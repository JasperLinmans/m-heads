from digitalpathologyutils.digitalpathologygenerators import DigitalPathologyGenerators
from digitalpathologyutils.digitalpathologydataloader import DigitalPathologyDataLoader
from utils.numpy_dataset import NumpyBatchDataset
import torch


def get_generators(config, state, data_path, param_path, x_files, y_files, x_val_files, y_val_files, create_stats, 
                   copy_directive, data_overrides, num_workers, generator_flag, copy_patches, logger):

    # Initialize batchgenerator from the library.
    # 
    GeneratorObject = DigitalPathologyGenerators(data_file_path=data_path,
                                                 param_file_path=param_path,
                                                 create_stats=create_stats,
                                                 data_copy_config=copy_directive,
                                                 data_overrides=data_overrides)

    if generator_flag:

        train_gen, valid_gen = GeneratorObject.get_generators()

        # Initialize data loader objects, used to define each epoch.
        #
        train_loader = DigitalPathologyDataLoader(batch_generator=train_gen,
                                                  repetition_count=config.training['iterations']['training']['repetition count'],
                                                  iter_count=config.training['iterations']['training']['iteration count'],
                                                  batch_size=config.training['iterations']['training']['batch size'])
        valid_loader = DigitalPathologyDataLoader(batch_generator=valid_gen,
                                                  repetition_count=config.training['iterations']['validation']['repetition count'],
                                                  iter_count=config.training['iterations']['validation']['iteration count'],
                                                  batch_size=config.training['iterations']['validation']['batch size'])           

        # Enable both training and validation data loaders to ping each other.
        #
        train_loader.set_generator_to_ping(valid_loader)
        valid_loader.set_generator_to_ping(train_loader)   
        
        return train_loader, valid_loader
    
    else:

        train_adapter, valid_adapter = GeneratorObject.get_adapters()

        GeneratorObject.training_generator.stop()
        GeneratorObject.validation_generator.stop()
    
        if config.model['type'] == 'deepensemble' and config.model['ensemble mode'] == 'single' and int(config.model['specific member']) < 99: 
            bootstrap_seed = config.model['specific member']
            logger.info('Going to use bootstrap seed: {}'.format(bootstrap_seed))
        else:
            bootstrap_seed = None

        train_dataset = NumpyBatchDataset(x_files, y_files, transform=train_adapter, bootstrap_seed=bootstrap_seed, 
                                          copy_flag=copy_patches, logger=logger)
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=num_workers, pin_memory=True,
                                                   batch_size=config.training['iterations']['training']['batch size'])

        validation_dataset = NumpyBatchDataset(x_val_files, y_val_files, transform=valid_adapter, copy_flag=copy_patches, logger=logger)
        valid_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, num_workers=num_workers, pin_memory=True,
                                                   batch_size=config.training['iterations']['validation']['batch size'])

        batches_per_epoch = config.training['iterations']['training']['repetition count'] * config.training['iterations']['training']['iteration count']
        train_loader.batches_per_epoch = batches_per_epoch

        batches_per_epoch = config.training['iterations']['validation']['repetition count'] * config.training['iterations']['validation']['iteration count']
        valid_loader.batches_per_epoch = batches_per_epoch

        # Continue with file of current checkpoint.
        #
        logger.info('Shifting numpy training files for {} epochs'.format(state['epoch']))
        for epoch in range(state['epoch']):
            train_loader.dataset.file_names = train_loader.dataset._shift(train_loader.dataset.file_names)
        if state['epoch']:
            train_loader.dataset._load_next()

        return train_loader, valid_loader
