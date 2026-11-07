"""Keep lightweight tests optional, but make learning verification explicit."""
import importlib

import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--require-learning', action='store_true',
        help='Fail if torch/torchvision cannot be imported instead of skipping learning tests.',
    )


def pytest_configure(config):
    if config.getoption('--require-learning'):
        for name in ('torch', 'torchvision'):
            try:
                importlib.import_module(name)
            except Exception as error:
                raise pytest.UsageError(
                    f'Learning verification requires a working {name} installation: {error}. '
                    'Run uv sync --project backend --extra learning --locked.'
                ) from error
