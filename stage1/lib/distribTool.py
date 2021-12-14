import torch.distributed as dist


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduceTensorMean(cfg, tensor):
    rt = tensor.clone()
    # print(rt)
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= cfg.envWorldSize

    # if 0 == cfg.envRank or 1 == cfg.envRank: print('rank={}, rtdivid={}'.format(cfg.envRank, rt))
    return rt

# def reduceTensorSum(tensor):
#     rt = tensor.clone()
#     dist.all_reduce(rt, op=dist.reduce_op.SUM)
#     return rt
