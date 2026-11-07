"""A missing learning dependency must not yield a green full verification run."""
import os
from pathlib import Path
import subprocess
import sys


def test_require_learning_rejects_missing_dependency():
    project = Path(__file__).resolve().parents[1]
    script = '''
import importlib.abc
import sys
import pytest
class MissingTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch':
            raise ModuleNotFoundError('simulated missing torch')
sys.meta_path.insert(0, MissingTorch())
raise SystemExit(pytest.main(['tests/test_simulation.py', '--require-learning', '-q']))
'''
    environment = os.environ | {'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
    result = subprocess.run([sys.executable, '-c', script], cwd=project, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'Learning verification requires a working torch installation' in result.stderr
    assert 'uv sync --project backend --extra learning --locked' in result.stderr
