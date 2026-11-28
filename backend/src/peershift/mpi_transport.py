"""Synchronous directed MPI snapshot transfers, with collective validation.

Every rank participates once per sequential round. Callers must abort the job
on failures outside collective validation; this is not fault-tolerant MPI.
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
        """Compatibility wrapper: receive one snapshot and scalar-peer metrics."""
        received, metrics = self.exchange_many(state, [selected_peer], round_index)
        metrics['selected_peer'] = metrics.pop('selected_peers')[0]
        metrics['received_sha256'] = metrics.pop('received_sha256s')[0]
        return received[0], metrics

    def exchange_many(self, state, peers, round_index):
        """Receive snapshots aligned with peers; serve all incoming requesters.

        peers must be a nonempty list/tuple of distinct other ranks. All local
        validation errors are announced collectively before posting payloads.
        No selection, discovery policy, or model aggregation occurs here.
        """
        MPI = self.MPI
        started = MPI.Wtime()
        packed = None
        metadata = dict(protocol_version=2, rank=self.rank, error=None)
        try:
            if self.size < 2:
                raise ValueError('MPI transport requires at least two ranks')
            if type(round_index) is not int or round_index != self.next_round:
                raise ValueError(f'Expected round {self.next_round}, got {round_index}')
            if (not isinstance(peers, (list, tuple)) or not peers
                    or any(type(peer) is not int or not 0 <= peer < self.size
                           or peer == self.rank for peer in peers)
                    or len(set(peers)) != len(peers)):
                raise ValueError('Selected peers must be distinct other valid ranks')
            peers = tuple(peers)
            packed = pack_state(state)
            if self.layout_id is not None and packed.layout_id != self.layout_id:
                raise ValueError('Model layout changed between rounds')
            metadata.update(round=round_index, selected_peers=peers,
                            dtype='float32', element_count=int(packed.values.size),
                            layout_id=packed.layout_id)
        except Exception as error:
            metadata['error'] = str(error)
        # Logical encoded JSON sizes exclude MPI framing/pickle/collective overhead.
        encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()
        metadata_started = MPI.Wtime()
        gathered = self.comm.allgather(encoded)
        metadata_seconds = MPI.Wtime()-metadata_started
        table = [json.loads(value) for value in gathered]
        failures = [f"rank {row['rank']}: {row['error']}" for row in table if row['error']]
        if failures:
            raise ValueError('MPI round validation failed: ' + '; '.join(failures))
        for rank, row in enumerate(table):
            requested = row['selected_peers']
            if (row['protocol_version'] != 2 or row['rank'] != rank
                    or row['round'] != round_index or row['dtype'] != 'float32'
                    or row['layout_id'] != packed.layout_id
                    or row['element_count'] != packed.values.size
                    or not isinstance(requested, list) or not requested
                    or any(type(peer) is not int or not 0 <= peer < self.size
                           or peer == rank for peer in requested)
                    or len(set(requested)) != len(requested)):
                raise ValueError('MPI round/layout/peer metadata mismatch')
        requesters = [rank for rank, row in enumerate(table) if self.rank in row['selected_peers']]
        received = [np.empty_like(packed.values) for _ in peers]
        # A private communicator, explicit sources and completed serial rounds
        # permit tag reuse without cross-round matching. Keep all buffers alive.
        try:
            payload_started = MPI.Wtime()
            requests = [self.comm.Irecv([buffer, MPI.FLOAT], source=peer, tag=0)
                        for peer, buffer in zip(peers, received)]
            requests.extend(self.comm.Isend([packed.values, MPI.FLOAT], dest=rank, tag=0)
                            for rank in requesters)
            statuses = [MPI.Status() for _ in requests]
            MPI.Request.Waitall(requests, statuses)
            if any(status.Get_count(MPI.FLOAT) != buffer.size
                   for status, buffer in zip(statuses, received)):
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
        buffers = [ModelBuffer(buffer, packed.layout) for buffer in received]
        metrics = dict(round=round_index, rank=self.rank, selected_peers=list(peers),
                       sent_model_messages=len(requesters), received_model_messages=len(peers),
                       sent_model_bytes=len(requesters)*packed.values.nbytes,
                       received_model_bytes=len(peers)*packed.values.nbytes,
                       metadata_local_encoded_bytes=len(encoded),
                       metadata_table_encoded_bytes=sum(map(len, gathered)),
                       metadata_collective_calls=1, synchronization_calls=1,
                       exchange_seconds=finished-started,
                       synchronization_seconds=finished-sync_started,
                       metadata_collective_seconds=metadata_seconds,
                       model_payload_seconds=sync_started-payload_started,
                       snapshot_sha256=packed.digest,
                       received_sha256s=[buffer.digest for buffer in buffers])
        return tuple(buffer.state_dict() for buffer in buffers), metrics
