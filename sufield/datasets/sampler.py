import torch

from ..lib.utils.distributed import get_rank, get_world_size


class InfSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, shuffle=True, seed=42) -> None:
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.it = 0
        self.gen_indices()

    def gen_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.indices = torch.randperm(len(self.data_source), generator=g).tolist()
        else:
            self.indices = torch.arange(len(self.data_source)).tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if self.it == len(self.data_source):
            self.it = 0
            self.epoch += 1
            self.gen_indices()
        value = self.indices[self.it]
        self.it += 1
        return value
    
    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedInfSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, shuffle=True, seed=42, drop_last=True) -> None:
        world_size = get_world_size()
        assert world_size > 1
        rank = get_rank()
        self.it = 0
        super(DistributedInfSampler, self).__init__(dataset, world_size, rank, shuffle, seed, drop_last)
        self.gen_indices()

    def gen_indices(self):
        self.indices = list(super().__iter__())

    def __iter__(self):
        return self

    def __next__(self):
        if self.it == self.num_samples:
            self.it = 0
            self.epoch += 1
            self.gen_indices()
        value = self.indices[self.it]
        self.it += 1
        return value
