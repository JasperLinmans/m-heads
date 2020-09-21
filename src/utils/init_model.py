import sys
sys.path.append('..')

from models.mheads import MHeads
from models.mcdropout import MCDropout
from models.deepensembles import DeepEnsemble


def get_model(config, multi_gpu):

    # Initialize, configure and build the model.
    #
    if config.model['type'] == 'mheads':
        model = MHeads()
        model.configure(input_shape=config.model['input shape'],
                        valid_padding=config.model['valid padding'],
                        pool_initial=config.model['pool initial'],
                        pool_final=config.model['pool final'],
                        layers_per_block=config.model['layers per block'],
                        growth_rate=config.model['growth rate'],
                        reduction=config.model['reduction'],
                        num_classes=config.model['num classes'],
                        bottleneck=config.model['bottleneck'],
                        dropout_rate=config.model['dropout rate'],
                        backbone=config.model['backbone'],
                        optimizer_type=config.model['optimizer type'],
                        optimizer_mom=config.model['optimizer mom'],
                        optimizer_wd=config.model['optimizer wd'],
                        head_depth=config.model['head depth'],
                        num_heads=config.model['num heads'],
                        mhead_random=config.model['mhead random'],
                        mhead_eps=config.model['mhead eps'],
                        learning_rate=config.model['learning rate'],
                        bottleneck_rate=config.model['bottleneck rate'],
                        multi_gpu=multi_gpu)
    elif config.model['type'] == 'mcdropout':
        model = MCDropout()
        model.configure(input_shape=config.model['input shape'],
                        valid_padding=config.model['valid padding'],
                        pool_initial=config.model['pool initial'],
                        pool_final=config.model['pool final'],
                        layers_per_block=config.model['layers per block'],
                        growth_rate=config.model['growth rate'],
                        reduction=config.model['reduction'],
                        num_classes=config.model['num classes'],
                        bottleneck=config.model['bottleneck'],
                        dropout_rate=config.model['dropout rate'],
                        backbone=config.model['backbone'],
                        optimizer_type=config.model['optimizer type'],
                        optimizer_mom=config.model['optimizer mom'],
                        optimizer_wd=config.model['optimizer wd'],
                        learning_rate=config.model['learning rate'],
                        num_samples=config.model['mc samples'],
                        bottleneck_rate=config.model['bottleneck rate'],
                        multi_gpu=multi_gpu)
    elif config.model['type'] == 'deepensemble':
        model = DeepEnsemble()
        model.configure(input_shape=config.model['input shape'],
                        valid_padding=config.model['valid padding'],
                        pool_initial=config.model['pool initial'],
                        pool_final=config.model['pool final'],
                        layers_per_block=config.model['layers per block'],
                        growth_rate=config.model['growth rate'],
                        reduction=config.model['reduction'],
                        num_classes=config.model['num classes'],
                        bottleneck=config.model['bottleneck'],
                        dropout_rate=config.model['dropout rate'],
                        backbone=config.model['backbone'],
                        optimizer_type=config.model['optimizer type'],
                        optimizer_mom=config.model['optimizer mom'],
                        optimizer_wd=config.model['optimizer wd'],
                        learning_rate=config.model['learning rate'],
                        num_members=config.model['num members'],
                        specific_member=config.model['specific member'],
                        ensemble_mode=config.model['ensemble mode'],
                        bottleneck_rate=config.model['bottleneck rate'],
                        multi_gpu=multi_gpu)
    else:
        raise ValueError('Incorrect model type given: {}'.format(config.model['type']))

    model.build()

    return model

