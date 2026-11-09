"""Real launcher tests: no mock communicator can substitute for these checks."""
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytest.importorskip('torch')
pytest.importorskip('mpi4py')
if shutil.which('mpiexec') is None:
    pytest.skip('MPI launcher not installed; use --require-mpi for mandatory verification', allow_module_level=True)
pytestmark = pytest.mark.mpi
PROJECT = Path(__file__).resolve().parents[1]


def launch(script, *args, ranks=3, timeout=45):
    command = ['mpiexec', '-n', str(ranks), sys.executable, '-c', script, *map(str, args)]
    # A launcher timeout must kill the complete rank process group, not only mpiexec.
    with subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


@pytest.mark.parametrize('pattern', ['ring', 'many_to_one', 'rotating'])
def test_demo_matches_frozen_snapshot_reference(tmp_path, pattern):
    config = tmp_path/'config.toml'
    config.write_text(f'pattern = "{pattern}"\nrounds = 3\n')
    script = 'from peershift.cli import main; import sys; sys.argv=["peershift", "mpi-demo", "--config", sys.argv[1], "--output", sys.argv[2]]; main()'
    for name in ('a', 'b'):
        result = launch(script, config, tmp_path/name)
        assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (tmp_path/'a/exchanges.jsonl').read_text().splitlines()]
    assert len(rows) == 9
    import torch
    from peershift.learning import make_model
    from peershift.model_buffer import pack_state
    reference = make_model(8)
    expected_bytes = sum(t.numel()*t.element_size() for t in reference.state_dict().values())
    expected = [1., 2., 3.]
    for round_index in range(3):
        current = rows[round_index*3:(round_index+1)*3]
        if pattern == 'ring':
            peers = [1, 2, 0]
        elif pattern == 'many_to_one':
            peers = [1, 0, 0]
        else:
            peers = [(rank+1+round_index%2)%3 for rank in range(3)]
        expected = [.5*value+.5*expected[peers[rank]] for rank, value in enumerate(expected)]
        assert [row['selected_peer'] for row in current] == peers
        assert [row['first_parameter_value'] for row in current] == expected
        for row in current:
            sender = current[row['selected_peer']]
            expected_state = {key: torch.full_like(tensor, expected[row['rank']])
                              for key, tensor in reference.state_dict().items()}
            assert row['averaged_sha256'] == pack_state(expected_state).digest
            assert row['received_model_bytes'] == expected_bytes
            assert row['received_sha256'] == sender['snapshot_sha256']
            assert row['source_snapshot_unchanged']
            assert row['sent_model_messages'] == peers.count(row['rank'])
            assert row['received_model_messages'] == 1
            assert row['sent_model_bytes'] == row['sent_model_messages']*row['received_model_bytes']
            assert row['exchange_seconds'] >= row['synchronization_seconds'] >= 0
    for name in ('a', 'b'):
        summary = json.loads((tmp_path/name/'summary.json').read_text())
        assert summary['sent_model_messages'] == summary['received_model_messages'] == 9
        assert summary['sent_model_bytes'] == summary['received_model_bytes']
    repeat = [json.loads(line) for line in (tmp_path/'b/exchanges.jsonl').read_text().splitlines()]
    timings = ('exchange_seconds', 'synchronization_seconds')
    # Logging sizes include encoded timing strings, so are also timing-dependent.
    ignored = (*timings, 'logging_local_encoded_bytes')
    assert [{k: v for k, v in row.items() if k not in ignored} for row in rows] == [
        {k: v for k, v in row.items() if k not in ignored} for row in repeat]
    rejected = launch(script, config, tmp_path/'a')
    assert rejected.returncode != 0 and 'Output directory must be empty' in rejected.stderr


@pytest.mark.parametrize('fault', ['peer', 'dtype', 'layout', 'round', 'nonfinite'])
def test_invalid_rank_input_rejected_collectively(tmp_path, fault):
    script = '''
import sys, torch
from mpi4py import MPI
from peershift.mpi_transport import MPITransport
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
transport=MPITransport(comm)
state={'x':torch.ones(4)}
peer=(rank+1)%comm.Get_size()
round_index=0
if rank==0:
    if sys.argv[1]=='peer': peer=rank
    if sys.argv[1]=='dtype': state={'x':torch.ones(4,dtype=torch.float64)}
    if sys.argv[1]=='layout': state={'different':torch.ones(4)}
    if sys.argv[1]=='round': round_index=1
    if sys.argv[1]=='nonfinite': state={'x':torch.full((4,),float('nan'))}
try:
    transport.exchange(state,peer,round_index)
except ValueError:
    print('REJECTED',rank,flush=True)
else:
    comm.Abort(2)
transport.close()
'''
    result = launch(script, fault)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('REJECTED') == 3


def test_large_payload_multiple_requesters_and_repeated_rounds():
    script = '''
import torch
from mpi4py import MPI
from peershift.mpi_transport import MPITransport
rank=MPI.COMM_WORLD.Get_rank()
transport=MPITransport()
peer=1 if rank==0 else 0
for round_index in range(2):
    state={'x':torch.full((600000,),float(rank+10*round_index))}
    received,metrics=transport.exchange(state,peer,round_index)
    assert torch.all(received['x']==peer+10*round_index)
    assert torch.all(state['x']==rank+10*round_index)
    assert metrics['received_model_bytes']==2400000
transport.close()
'''
    result = launch(script)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('ranks', [1, 2])
def test_demo_rejects_rank_count_mismatch(tmp_path, ranks):
    config = tmp_path/'config.toml'
    config.write_text('num_clients=3\nrounds=1\n')
    script = 'from peershift.cli import main; import sys; sys.argv=["peershift", "mpi-demo", "--config", sys.argv[1], "--output", sys.argv[2]]; main()'
    result = launch(script, config, tmp_path/'output', ranks=ranks)
    assert result.returncode != 0
    assert 'MPI rank count must equal num_clients' in result.stderr


def test_layout_change_after_successful_round_is_collectively_rejected():
    script = """
import torch
from mpi4py import MPI
from peershift.mpi_transport import MPITransport
transport=MPITransport()
rank=transport.rank
peer=(rank+1)%transport.size
transport.exchange({'x':torch.ones(4)},peer,0)
state={'x':torch.ones(4)} if rank else {'x':torch.ones(2,2)}
try:
    transport.exchange(state,peer,1)
except ValueError:
    print('REJECTED',rank,flush=True)
else:
    MPI.COMM_WORLD.Abort(2)
transport.close()
"""
    result = launch(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('REJECTED') == 3


def test_timeout_kills_launcher_process_group(monkeypatch):
    class HangingLauncher:
        pid = 12345
        returncode = -signal.SIGKILL
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired('mpiexec', timeout)
            return '', 'ranks terminated'

    options, killed = {}, []
    def spawn(command, **kwargs):
        options.update(kwargs)
        return HangingLauncher()
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    monkeypatch.setattr(os, 'killpg', lambda pid, sig: killed.append((pid,sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        launch('unused', timeout=.01)
    assert options['start_new_session'] is True
    assert killed == [(12345, signal.SIGKILL)]
