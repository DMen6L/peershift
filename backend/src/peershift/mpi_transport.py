"""Collective round control and directed point-to-point model transport.

Every communicator rank must call exchange once per round, in order. This is
synchronous transport, not fault-tolerant MPI. The caller must abort the job if
one rank fails outside collective validation. No peer discovery is performed.
"""
import json
import sys

import numpy as np

from .model_buffer import ModelBuffer, pack_state


class MPITransport:
    def __init__(self, comm=None):
        from mpi4py import MPI
        self.MPI = MPI
        self.comm = (MPI.COMM_WORLD if comm is None else comm).Dup()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.next_round = 0
        self.layout_id = None

    def close(self):
        """Collective: call only once all ranks have finished using the transport."""
        self.comm.Free()

    def exchange(self, state, selected_peer, round_index):
        """Receive one peer snapshot and serve every requester of our snapshot.

        Returns (received_state_dict, per_rank_metrics). All local input errors
        are announced collectively before payload requests are posted.
        """
        MPI = self.MPI
        started = MPI.Wtime()
        packed = None
        metadata = dict(protocol_version=1, rank=self.rank, error=None)
        try:
            if self.size < 2:
                raise ValueError('MPI transport requires at least two ranks')
            if type(round_index) is not int or round_index != self.next_round:
                raise ValueError(f'Expected round {self.next_round}, got {round_index}')
            if (type(selected_peer) is not int or not 0 <= selected_peer < self.size
                    or selected_peer == self.rank):
                raise ValueError('Selected peer must be another valid rank')
            packed = pack_state(state)
            if self.layout_id is not None and packed.layout_id != self.layout_id:
                raise ValueError('Model layout changed between rounds')
            metadata.update(round=round_index, selected_peer=selected_peer,
                            dtype='float32', element_count=int(packed.values.size),
                            layout_id=packed.layout_id)
        except Exception as error:
            metadata['error'] = str(error)
        # These byte counts describe the logical JSON records, not MPI framing,
        # pickle overhead, or a collective algorithm's physical network traffic.
        encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()
        gathered = self.comm.allgather(encoded)
        table = [json.loads(value) for value in gathered]
        failures = [f"rank {row['rank']}: {row['error']}" for row in table if row['error']]
        if failures:
            raise ValueError('MPI round validation failed: ' + '; '.join(failures))
        for rank, row in enumerate(table):
            if (row['protocol_version'] != 1 or row['rank'] != rank
                    or row['round'] != round_index or row['dtype'] != 'float32'
                    or row['layout_id'] != packed.layout_id
                    or row['element_count'] != packed.values.size
                    or type(row['selected_peer']) is not int
                    or not 0 <= row['selected_peer'] < self.size
                    or row['selected_peer'] == rank):
                raise ValueError('MPI round/layout/peer metadata mismatch')
        requesters = [rank for rank, row in enumerate(table) if row['selected_peer'] == self.rank]
        received = np.empty_like(packed.values)
        # A private communicator, explicit sources, and completed serial rounds
        # permit reuse of tag zero without cross-round matching.
        try:
            requests = [self.comm.Irecv([received, MPI.FLOAT], source=selected_peer, tag=0)]
            requests.extend(self.comm.Isend([packed.values, MPI.FLOAT], dest=rank, tag=0)
                            for rank in requesters)
            statuses = [MPI.Status() for _ in requests]
            MPI.Request.Waitall(requests, statuses)
            if statuses[0].Get_count(MPI.FLOAT) != received.size:
                raise RuntimeError('Received model payload has the wrong length')
            sync_started = MPI.Wtime()
            self.comm.Barrier()
        except Exception as error:
            print(f'Rank {self.rank}: MPI payload failure: {error}', file=sys.stderr, flush=True)
            self.comm.Abort(1)
            raise
        finished = MPI.Wtime()
        self.next_round += 1
        self.layout_id = packed.layout_id
        metrics = dict(round=round_index, rank=self.rank, selected_peer=selected_peer,
                       sent_model_messages=len(requesters), received_model_messages=1,
                       sent_model_bytes=len(requesters)*packed.values.nbytes,
                       received_model_bytes=received.nbytes,
                       metadata_local_encoded_bytes=len(encoded),
                       metadata_table_encoded_bytes=sum(map(len, gathered)),
                       metadata_collective_calls=1, synchronization_calls=1,
                       exchange_seconds=finished-started,
                       synchronization_seconds=finished-sync_started,
                       snapshot_sha256=packed.digest,
                       received_sha256=ModelBuffer(received, packed.layout).digest)
        return ModelBuffer(received, packed.layout).state_dict(), metrics
