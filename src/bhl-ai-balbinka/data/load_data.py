""" This module helps locate directories in the main project directory.
"""
from pathlib import Path
from filter_data import filter_data

# path for the project directory
project_dir = Path(__file__).resolve().parents[3]


def get_data_dir() -> Path:
    """
    Gets raw directory path.
    """
    return project_dir / 'data'


def get_results_dir() -> Path:
    """
    Gets results directory path.
    """
    return project_dir / 'results'


def get_reports_dir() -> Path:
    """
    Gets reports directory path.
    """
    return project_dir / 'reports'


def get_references_dir() -> Path:
    """
    Gets references directory path.
    """
    return project_dir / 'references'


def main_load():
    training_path = get_data_dir() / 'SDOBenchmark-data-example' / 'training'
    folders = filter_data(training_path)
    print(folders)


main_load()
