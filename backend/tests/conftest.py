"""Keep lightweight tests optional, but make learning verification explicit."""
import importlib
import shutil

import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--require-learning', action='store_true',
        help='Fail if torch/torchvision cannot be imported instead of skipping learning tests.',
    )
    parser.addoption(
        '--require-mpi', action='store_true',
        help='Require MPI dependencies and launcher for real multi-process transport tests.',
    )


def pytest_configure(config):
    config.addinivalue_line('markers', 'mpi: real multi-process MPI integration tests')
    if config.getoption('--require-mpi'):
        for name in ('numpy', 'torch', 'mpi4py.MPI'):
            try:
                importlib.import_module(name)
            except Exception as error:
                raise pytest.UsageError(
                    f'MPI verification requires a working {name}: {error}. '
                    'Install an MPI runtime and run uv sync --project backend --extra mpi --locked.'
                ) from error
        if shutil.which('mpiexec') is None:
            raise pytest.UsageError('MPI verification requires mpiexec from a supported MPI runtime.')
    if config.getoption('--require-learning'):
        for name in ('torch', 'torchvision'):
            try:
                importlib.import_module(name)
            except Exception as error:
                raise pytest.UsageError(
                    f'Learning verification requires a working {name} installation: {error}. '
                    'Run uv sync --project backend --extra learning --locked.'
                ) from error
