# -*- coding: utf-8 -*-
"""Tensor-parallel process group.

One process per GPU. Rank r owns heads [r*H/tp, (r+1)*H/tp) of every time-mix and
columns [r*F/tp, (r+1)*F/tp) of every channel-mix; the residual stream is
replicated. Two all-reduces per layer (time-mix output, channel-mix output), the
Megatron pattern.

NCCL notes for GPUs without peer access (consumer cards on PCIe): NCCL goes
through host shared memory ("SHM/direct"), and with cuMem allocations that
transport hangs on the first collective (torch 2.14 / NCCL 2.30). NCCL_CUMEM_ENABLE=0
fixes it; it is set here unless the environment already decides. Measured on two
PCIe GPUs without P2P: ~24 us per all-reduce of one 8 KB row inside a CUDA graph,
~53 us at 16 rows.

Keep the ranks in lockstep. A rank that runs many steps ahead (no host sync) through a mix of CUDA-graph
replays with collectives inside and eager collectives was seen to hang over SHM (tools/test_tp.py before it
synced per step). The engine is lockstep by construction: rank 0 syncs every step to sample, and the workers
only get the next command through the channel after that.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


class TP:
    """rank / world of the tensor-parallel group (world 1 = no communication at all)."""

    def __init__(self, rank: int = 0, world: int = 1):
        self.rank, self.world = rank, world

    @property
    def enabled(self):
        return self.world > 1

    def all_reduce(self, x):
        if self.world > 1:
            dist.all_reduce(x)
        return x

    def broadcast(self, x, src=0):
        if self.world > 1:
            dist.broadcast(x, src)
        return x

    def barrier(self):
        if self.world > 1:
            dist.barrier()


def init(rank: int, world: int, port: int, device_index: int) -> TP:
    torch.cuda.set_device(device_index)
    if world > 1:
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                                device_id=torch.device("cuda", device_index))
    return TP(rank, world)


def shutdown(tp: TP):
    """Nothing to do: processes exit with os._exit. destroy_process_group() over the SHM
    transport hangs at teardown (seen with NCCL 2.30), and the process is going away anyway."""
    return None
