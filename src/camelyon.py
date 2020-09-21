from digitalpathologyutils.digitalpathologyconfiguration import DigitalPathologyConfiguration
from digitalpathologyutils.digitalpathologyexperimentcommander import ExperimentCommander
from digitalpathologyutils.digitalpathologystate import DigitalPathologyState
from utils.log_head_report import init_head_report, log_normalized_head_report
from utils.argparse_camelyon import collect_arguments
from utils.init_model import get_model
from utils.init_generators import get_generators
from digitalpathology.utils import trace as dpttrace
from collections import Counter
import neptune
from neptune.sessions import Session
import sys
import os
import numpy as np
import traceback
import torch
from torchsummary import summary
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
import random
from tqdm import tqdm

def run():

    # Parse arguments.
    #
    (experiment_name,
     param_path,
     work_dir,
     data_path,
     copy_directive,
     data_overrides,
     data_root_dir,
     x_files, 
     y_files,
     x_val_files,
     y_val_files,
     repo_dir,
     param_overrides,
     log_level,
     random_seed,
     num_workers,
     continue_experiment,
     copy_patches,
     multi_gpu,
     create_stats,
     generator_flag,
     neptune_project) = collect_arguments()
    
    ###################################################################################
    # Initialization: Initialize logging, data-loaders and model.
    ###################################################################################

    # Extend experiment commander with utility functions.
    #
    ExperimentCommander.init_head_report = init_head_report
    ExperimentCommander.log_normalized_head_report = log_normalized_head_report

    # Initialize experiment commander object.
    #
    cmd = ExperimentCommander(experiment_name=experiment_name,
                              work_dir_path=work_dir,
                              data_file_path=None,
                              data_overrides=data_overrides,
                              data_copy_config=copy_directive,
                              param_file_path=param_path,
                              param_overrides=param_overrides,
                              create_stats=create_stats,
                              continue_experiment=continue_experiment,
                              repository_root_dir=repo_dir,
                              file_log_level=log_level,
                              seed=random_seed)

    # Create config object from configuration file.
    #
    config = DigitalPathologyConfiguration(cmd.get_parameter_path(), param_overrides=cmd.get_param_overrides())

    # Set up neptune experiment.
    #
    if cmd.existing_neptune_experiment() and continue_experiment:

        project_name, experiment_id = cmd.get_experiment_project_id()
        
        session = Session()  # with_default_backend()
        project = session.get_project(project_qualified_name=project_name)
        experiment = project.get_experiments(id=experiment_id)[0]
        cmd.logger.info('Succesfully loaded an existing neptune experiment.')
        
    else:
        cmd.logger.info('Initialising neptune experiment.')
        neptune.init(project_qualified_name=neptune_project)
        experiment = neptune.create_experiment(name=experiment_name,
                                               params=config.get_complete_config_dict(),
                                               logger=cmd.logger)
        cmd.save_experiment_project_id(neptune_project, experiment.id)

    # Initialize, configure and build the model.
    #
    model = get_model(config, multi_gpu)
    cmd.logger.info('Done initialising {} model.'.format(config.model['type']))

    # Print model summary.
    #
    if config.model['type'] == 'deepensemble' and config.model['ensemble mode'] != 'single':
        summary(model._model_instance[0], input_size=[config.model['input shape']])
    else:
        summary(model._model_instance, input_size=[config.model['input shape']])

    # Initialize state.
    #
    state = DigitalPathologyState(plateau_metric=config.training['metric name'],
                                  plateau_metric_higher_is_better=config.training['higher is better'],
                                  plateau_averaging_length=config.training['averaging length'])

    # Try to load previous checkpoint, if exists.
    #
    try:
        cmd.load(model, state)
        cmd.logger.info('Succesfully loaded a previous checkpoint.')
    except FileNotFoundError:
        cmd.logger.info('No previous checkpoint available.')
        pass
   
    # Initialize training and validation dataloaders.
    #
    train_loader, valid_loader = get_generators(config=config,
                                                state=state,
                                                data_path=data_path,
                                                param_path=param_path,
                                                x_files=x_files,
                                                y_files=y_files,
                                                x_val_files=x_val_files,
                                                y_val_files=y_val_files,
                                                create_stats=create_stats,
                                                copy_directive=copy_directive,
                                                data_overrides=data_overrides,
                                                num_workers=num_workers,
                                                generator_flag=generator_flag,
                                                copy_patches=copy_patches,
                                                logger=cmd.logger)

    # Initiate csv logger.
    #
    cmd.init_csv_logger(model.metricnames())

    # Initialize head report.
    #
    cmd.init_head_report()

    ###################################################################################
    # Training stage: Train and validate model.
    ###################################################################################
    
    cmd.logger.info('Initialising training.')
    for epoch in range(state['epoch'], config.model['epochs']):

        # Update learning rate based on predetermined scheme.
        #
        if epoch in config.model['learning rate scheme']:
            cmd.logger.info('Going to update learning rate to: {}'.format(config.model['learning rate scheme'][epoch]))
            model.updatelearningrate(config.model['learning rate scheme'][epoch])

        # Training epoch.
        #
        training_stats = Counter() 
        for i, (data, target) in enumerate(train_loader):
            
            if isinstance(train_loader, torch.utils.data.DataLoader):

                # Manually randomize augmentation pool.
                #
                train_loader.dataset.transform_object._BatchAdapter__augmenter_pool.randomize() 
                
                data, target = data.numpy(), target.numpy()
                train_loader.dataset.running_idx += train_loader.batch_size

                if train_loader.dataset.running_idx >= train_loader.dataset.buffer_size:
                    cmd.logger.info('Running index of train loader is: {}, updating buffer...'.format(train_loader.dataset.running_idx))
                    train_loader.dataset._load_next()

            output = model.update(data, target)
            training_stats.update(output)
            
            if i % 100  == 0:
                cmd.logger.info('Epoch: {}, training step: {} / {}'.format(epoch, i, train_loader.batches_per_epoch))

            if i >= train_loader.batches_per_epoch - 1:
                break

        training_steps = train_loader.batches_per_epoch
        training_stats = {k: v / training_steps for k, v, in training_stats.items()}

        # Report on epoch stats.
        #
        info_str = ', '.join(['{}: {:.3f}'.format(k, training_stats[k]) for k in model.metricnames()])
        cmd.logger.info('Training: ' + info_str)
        for stat, value in training_stats.items():
            experiment.send_metric('train_'+stat, value)

        # Validation epoch.
        #
        validation_stats = Counter() 
        for i, (data, target) in enumerate(valid_loader):

            if isinstance(valid_loader, torch.utils.data.DataLoader):
                data, target = data.numpy(), target.numpy()

            output = model.validate(data, target)
            validation_stats.update(output)

            if i % 100 == 0:
                cmd.logger.info('Epoch: {}, validation step: {} / {}'.format(epoch, i, valid_loader.batches_per_epoch))

            if i >= valid_loader.batches_per_epoch - 1:
                break

        validation_steps = valid_loader.batches_per_epoch
        validation_stats = {k: v / validation_steps for k, v, in validation_stats.items()}

        # Report on epoch stats.
        #
        info_str = ', '.join(['{}: {:.3f}'.format(k, validation_stats[k]) for k in model.metricnames()])
        cmd.logger.info('Validation: ' + info_str)
        for stat, value in validation_stats.items():
            experiment.send_metric('valid_'+stat, value)

        # Write output to csv file.
        #
        csv_output = [training_stats[k] for k in model.metricnames()] + [validation_stats[k] for k in model.metricnames()]
        cmd.log_csv_row(csv_output)

        # Normalize head report.
        #
        if config.model['type'] == 'mheads':
            for purpose in model.head_report:
                total = sum(model.head_report[purpose].values())
                model.head_report[purpose] = [model.head_report[purpose][i] / total for i in range(config.model['num heads'])]

            # Log head report and reset counter.
            #
            cmd.log_normalized_head_report(model.head_report)
            experiment.send_text('training head report', x=epoch, y=str(model.head_report['training']))
            experiment.send_text('validation head report', x=epoch, y=str(model.head_report['validation']))
            model.reset_head_report()

        # Update state based on validation stats for checkpointing and early stopping.
        #
        _ = state.update_state(validation_stats)
        
        # Increment epoch counter.
        #
        state.increment_epoch()

        # Save state and model.
        #
        if state.improved():
            cmd.save(model, state, best=True)
        cmd.save(model, state, best=False)

    # Report best validation metric and index of the epoch.
    #
    cmd.logger.info('Best validation {metric} was {value} at epoch {index}'.format(metric=config.training['metric name'],
                                                                                   value=state['best_plateau_metric'],
                                                                                   index=state['best_plateau_metric_epoch']))

    experiment.stop()

if __name__ == '__main__':

    try:
        run()    

    except Exception as exception:

        print('Original stacktrace:')
        print(traceback.print_exc(file=sys.stdout))
        print()

        # Collect and summarize traceback information.
        #
        _, _, exception_traceback = sys.exc_info()
        trace_string = dpttrace.format_traceback(traceback_object=exception_traceback)

        print(exception.__class__, exception)
        print(trace_string)

        _, _, exception_traceback = sys.exc_info()
        neptune.stop(traceback=exception_traceback)

