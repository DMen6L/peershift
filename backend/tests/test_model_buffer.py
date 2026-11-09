import pytest

torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')
from peershift.model_buffer import pack_state


def test_snapshot_round_trip_and_independent_storage():
    state = {'z': torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
             'a': torch.tensor(7., dtype=torch.float32)}
    packed = pack_state(state)
    original_digest = packed.digest
    restored = packed.state_dict()
    assert packed.values.flags.c_contiguous and not packed.values.flags.writeable
    assert [name for name, _ in packed.layout] == ['a', 'z']
    assert packed.layout_id == pack_state(dict(reversed(list(state.items())))).layout_id
    for name, tensor in state.items():
        assert torch.equal(tensor, restored[name])
        assert tensor.data_ptr() != restored[name].data_ptr()
    state['z'].zero_()
    restored['a'].fill_(0)
    assert packed.digest == original_digest
    assert packed.state_dict()['a'].item() == 7


@pytest.mark.parametrize('state', [{}, {'x': torch.ones(2, dtype=torch.float64)},
                                  {'x': torch.ones(2, dtype=torch.int64)},
                                  {'x': torch.tensor([float('nan')])},
                                  {'x': torch.tensor([float('inf')])},
                                  {'x': torch.empty(0)}])
def test_snapshot_rejects_unsupported_data(state):
    with pytest.raises(ValueError):
        pack_state(state)


def test_layout_hash_covers_names_and_shapes():
    baseline = pack_state({'x': torch.ones(4)})
    assert baseline.layout_id != pack_state({'y': torch.ones(4)}).layout_id
    assert baseline.layout_id != pack_state({'x': torch.ones(2, 2)}).layout_id
