from torch.utils.data import Sampler
import torch.distributed as dist
import math


class DistributedSamplerVali(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        # super(DistributedSamplerVali, self).__init__()
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch

        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class Sample(object):
    __slots__ = {'I0', 'I1', 'It', 'Et', 'I0t', 'I1t'}

    def __init__(self, I0, I1, It, Et, I0t, I1t):
        super(Sample, self).__init__()

        self.I0 = I0
        self.I1 = I1
        self.It = It
        self.Et = Et

        self.I0t = I0t
        self.I1t = I1t


class Samples(object):
    __slots__ = {'I1', 'I2', 'I3', 'I4', 'I5',
                 'E2', 'E3', 'E4',
                 'I12', 'I13', 'I14', 'I23', 'I24', 'I32', 'I34', 'I43', 'I42', 'I54', 'I53', 'I52'}

    def __init__(self, I1, I2, I3, I4, I5,
                 E2, E3, E4,
                 I12, I13, I14, I23, I24, I32, I34, I43, I42, I54, I53, I52):
        super(Samples, self).__init__()
        self.I1, self.I2, self.I3, self.I4, self.I5 = I1, I2, I3, I4, I5

        self.E2, self.E3, self.E4 = E2, E3, E4

        self.I12, self.I13, self.I14 = I12, I13, I14

        self.I23, self.I24 = I23, I24
        self.I32, self.I34 = I32, I34
        self.I43, self.I42 = I43, I42
        self.I54, self.I53, self.I52 = I54, I53, I52

    def getSample(self, idx):
        cases = {'0': Sample(self.I1, self.I3, self.I2, self.E2, self.I12, self.I32),

                 '1': Sample(self.I1, self.I4, self.I2, self.E2, self.I12, self.I42),

                 '2': Sample(self.I1, self.I5, self.I2, self.E2, self.I12, self.I52),

                 '3': Sample(self.I1, self.I4, self.I3, self.E3, self.I13, self.I43),

                 '4': Sample(self.I1, self.I5, self.I3, self.E3, self.I13, self.I53),

                 '5': Sample(self.I1, self.I5, self.I4, self.E4, self.I14, self.I54),

                 '6': Sample(self.I2, self.I4, self.I3, self.E3, self.I23, self.I43),

                 '7': Sample(self.I2, self.I5, self.I3, self.E3, self.I23, self.I53),

                 '8': Sample(self.I2, self.I5, self.I4, self.E4, self.I24, self.I54),

                 '9': Sample(self.I3, self.I5, self.I4, self.E4, self.I34, self.I54)
                 }
        return cases[str(idx)]

