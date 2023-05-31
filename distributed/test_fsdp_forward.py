import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
import torch.multiprocessing as mp

import argparse
import os
from copy import deepcopy

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Model(Module):
    def __init__(self, wrap_fsdp):
        super().__init__()
        torch.manual_seed(0)
        self.inner = Linear(4, 4)
        if wrap_fsdp:
            self.inner = FSDP(self.inner)
        self.outer = Linear(4, 5)
    
    def forward(self, x):
        i = self.inner(x)
        j = self.inner(x)
        return self.outer(i + j)

def _dist_train(wrap_fsdp):
    torch.manual_seed(0)
    model = Model(wrap_fsdp).cuda()
    if wrap_fsdp:
        model = FSDP(model)
    else:
        model = DDP(model)
    optim = SGD(model.parameters(), lr=0.1)
    
    in_data = torch.randn(64, 4).cuda()
    in_data.requires_grad = True
    for _ in range(3):
        out = model(in_data)
        out.sum().backward()
        optim.step()
        optim.zero_grad()

    if wrap_fsdp:
        #return get_full_params(model)
        with FSDP.summon_full_params(model, recurse=True):
            return deepcopy(list(model.parameters()))

    return list(model.parameters())

def test_multi_fwd(rank, world_size, args):
    torch.cuda.set_device(rank)
    print ("Running on local rank : {}".format(rank))
    setup(rank, world_size)
    ddp_state = _dist_train(wrap_fsdp=False)
    fsdp_state = _dist_train(wrap_fsdp=True)
    #torch.allclose(
    #print (ddp_state)
    #print (fsdp_state)
    total_count = 0
    for ddp_p, fsdp_p in zip(ddp_state, fsdp_state):
        result = torch.allclose(ddp_p.data, fsdp_p.data, atol=1e-05, rtol=1e-06) 
        if not result:
            total_count = total_count + 1
            print ("------ DDP value ------ ")
            print (ddp_p.data)
            print ("-------- FSDP value ------ ")
            print (fsdp_p.data)
    print ("Rank : {}, total_failed_tensors : {}".format(rank, total_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_nodes', default=1, type=int, help='total number of the nodes')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--node_rank', default=0, type=int, help='rank of present node (server).')
    args = parser.parse_args() ## just dummy to pass the args in mp.spawn()
    
    
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(test_multi_fwd,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
