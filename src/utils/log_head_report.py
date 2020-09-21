"""
Extend the experiment commander class with a utility function to log the winning heads.
"""
import csv
import os

def init_head_report(self):
    """Create csv output file"""

    train_head_log_file_path = os.path.join(self._ExperimentCommander__log_output_dir, 'train_head_report.csv')
    valid_head_log_file_path = os.path.join(self._ExperimentCommander__log_output_dir, 'valid_head_report.csv')

    if os.path.isfile(train_head_log_file_path) and os.path.isfile(valid_head_log_file_path) and self._ExperimentCommander__continue_experiment:
        return

    open(train_head_log_file_path, 'w').close()
    open(valid_head_log_file_path, 'w').close()

def log_normalized_head_report(self, norm_head_report):
    """Write head report to csv files."""

    train_head_log_file_path = os.path.join(self._ExperimentCommander__log_output_dir, 'train_head_report.csv')
    valid_head_log_file_path = os.path.join(self._ExperimentCommander__log_output_dir, 'valid_head_report.csv')

    with open(train_head_log_file_path, 'a') as csv_file:
        csv_logger = csv.writer(csv_file)
        csv_logger.writerow(norm_head_report['training'])

    with open(valid_head_log_file_path, 'a') as csv_file:
        csv_logger = csv.writer(csv_file)
        csv_logger.writerow(norm_head_report['validation'])
