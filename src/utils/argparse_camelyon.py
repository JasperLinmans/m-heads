import argparse
import os


def none_or_str(value):
    if value.lower() == 'none':
        return None
    return value


def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        The parsed command line arguments.
    """

    # Prepare argument value choices.
    #
    logging_level_choices = ['debug', 'info', 'warning', 'error', 'critical']

    # Set project level output directory.
    #
    project_output = ''

    # Set data root directory.
    #
    data_root_path = ''

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Set up experiment for uncertainty estimation.')

    argument_parser.add_argument('-e',  '--experiment',          required=True,  type=str,                                                help='Experiment name')
    argument_parser.add_argument('-p',  '--parameters',          required=True,  type=str,                                                help='Input parameters file')
    argument_parser.add_argument('-d',  '--data',                required=False, type=none_or_str,                                        help='Input data file')
    argument_parser.add_argument('-cp', '--copy_from',           required=False, type=none_or_str, default=None,                          help='Data copy source')
    argument_parser.add_argument('-do', '--data_override',       required=False, type=none_or_str, default=None,                          help='Data source overrides')
    argument_parser.add_argument('-w',  '--work',                required=False, type=str, default=project_output,                        help='Work directory')
    argument_parser.add_argument('-dr', '--data_root',           required=False, type=str, default=data_root_path,                        help='Data root path directory')
    argument_parser.add_argument('-x',  '--x_files',             required=False, type=none_or_str, default=None,                             help='Declares patch files')
    argument_parser.add_argument('-y',  '--y_files',             required=False, type=none_or_str, default=None,                             help='Declares label files')
    argument_parser.add_argument('-xv', '--x_val_files',         required=False, type=none_or_str, default=None,                             help='Declares validation patch files')
    argument_parser.add_argument('-yv', '--y_val_files',         required=False, type=none_or_str, default=None,                             help='Declares validation label files')
    argument_parser.add_argument('-r',  '--repositories',        required=False, type=str, default='/home/user/source',                   help='Additional repositories to report')
    argument_parser.add_argument('-n',  '--neptune_project',     required=False, type=str, default='',               help='Neptune project used for logging')
    argument_parser.add_argument('-po', '--parameters_override', required=False, type=str, default=None,                                  help='Parameter value overrides')
    argument_parser.add_argument('-l',  '--logging_level',       required=False, type=str, default='info', choices=logging_level_choices, help='File logging level')
    argument_parser.add_argument('-s',  '--seed',                required=False, type=int, default=None,                                  help='Random seed')
    argument_parser.add_argument('-nw',  '--num_workers',         required=False, type=int, default=0,                                     help='Amount of workers used')
    argument_parser.add_argument('-c',  '--continue',            action='store_true',                                                     help='Continue experiment if possible')
    argument_parser.add_argument('-m',  '--multi_gpu',           action='store_true',                                                     help='Use multiple gpus')
    argument_parser.add_argument('-st',  '--stats',               action='store_true',                                                     help='Stat files')
    argument_parser.add_argument('-cpp',  '--copy_patches',               action='store_true',                                                     help='Copy patch dataset')
    argument_parser.add_argument('-g',  '--generator',           action='store_true',                                                     help='Use batch generator from library')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_experiment_name = arguments['experiment']
    parsed_param_path = arguments['parameters']
    parsed_work_dir_path = arguments['work']
    parsed_data_path = arguments['data']
    parsed_copy_directive_str = arguments['copy_from']
    parsed_data_override_map_str = arguments['data_override']
    parsed_data_root_dir_path = arguments['data_root']
    parsed_parameters_override_map_str = arguments['parameters_override']
    parsed_repositories_path = arguments['repositories']
    parsed_logging_level = arguments['logging_level']
    parsed_random_seed = arguments['seed']
    parsed_num_workers = arguments['num_workers']
    parsed_continue_flag = arguments['continue']
    parsed_stats_flag = arguments['stats']
    parsed_multi_gpu_flag = arguments['multi_gpu']
    parsed_copy_patches_flag = arguments['copy_patches']
    parsed_generator_flag = arguments['generator']
    parsed_neptune_project = arguments['neptune_project']
    parsed_x_files = arguments['x_files']
    parsed_y_files = arguments['y_files']
    parsed_x_val_files = arguments['x_val_files']
    parsed_y_val_files = arguments['y_val_files']
     
    # Parse numpy dataset files.
    # 
    parsed_x_files =     None if parsed_x_files is None else [os.path.join(parsed_data_root_dir_path, data_file) for data_file in parsed_x_files.split(' ')] 
    parsed_y_files =     None if parsed_y_files is None else [os.path.join(parsed_data_root_dir_path, data_file) for data_file in parsed_y_files.split(' ')]
    parsed_x_val_files = None if parsed_x_val_files is None else [os.path.join(parsed_data_root_dir_path, data_file) for data_file in parsed_x_val_files.split(' ')]
    parsed_y_val_files = None if parsed_y_val_files is None else [os.path.join(parsed_data_root_dir_path, data_file) for data_file in parsed_y_val_files.split(' ')]

    # Evaluate expressions
    #    
    parsed_copy_directive_map = eval(parsed_copy_directive_str) if parsed_copy_directive_str else None
    parsed_data_override_map = eval(parsed_data_override_map_str) if parsed_data_override_map_str else None
    parsed_parameters_override_map = eval(parsed_parameters_override_map_str) if parsed_parameters_override_map_str else None

    # Print parameters.
    #
    print(argument_parser.description)
    print('Experiment name: {name}'.format(name=parsed_experiment_name))
    print('Parameters file: {parameters}'.format(parameters=parsed_param_path))
    print('Work directory: {move}'.format(move=parsed_work_dir_path))
    print('Data file: {data_file}'.format(data_file=parsed_data_path))
    print('Copy source: {map}'.format(map=parsed_copy_directive_map))
    print('Data path overrides: {map}'.format(map=parsed_data_override_map))
    print('Data root directory: {move}'.format(move=parsed_data_root_dir_path))
    print('x files: {move}'.format(move=parsed_x_files))
    print('y files: {move}'.format(move=parsed_y_files))
    print('x validation files: {move}'.format(move=parsed_x_val_files))
    print('y validation files: {move}'.format(move=parsed_y_val_files))
    print('Repositories to report: {repos}'.format(repos=parsed_repositories_path))
    print('Parameter value overrides: {map}'.format(map=parsed_parameters_override_map))
    print('File logging level: {level}'.format(level=parsed_logging_level))
    print('Random seed: {seed}'.format(seed=parsed_random_seed))
    print('Number of workers: {workers}'.format(workers=parsed_num_workers))
    print('Continue experiment: {flag}'.format(flag=parsed_continue_flag))
    print('Multi GPU: {flag}'.format(flag=parsed_multi_gpu_flag))
    print('Copy patches flag: {flag}'.format(flag=parsed_copy_patches_flag))
    print('Stats: {flag}'.format(flag=parsed_stats_flag))
    print('Generator: {flag}'.format(flag=parsed_generator_flag))
    print('Logging to neptune project: {project}'.format(project=parsed_neptune_project))

    # Return parsed values.
    #
    return (parsed_experiment_name,
            parsed_param_path,
            parsed_work_dir_path,
            parsed_data_path,
            parsed_copy_directive_map,
            parsed_data_override_map,
            parsed_data_root_dir_path,
            parsed_x_files,
            parsed_y_files,
            parsed_x_val_files,
            parsed_y_val_files,
            parsed_repositories_path,
            parsed_parameters_override_map,
            parsed_logging_level,
            parsed_random_seed,
            parsed_num_workers,
            parsed_continue_flag,
            parsed_copy_patches_flag,
            parsed_multi_gpu_flag,
            parsed_stats_flag,
            parsed_generator_flag,
            parsed_neptune_project)

