from logging import getLogger

import pointnet2._ext as p2
import torch
from torch import logit, nn

from ..datasets import transforms as t
from ..lib.utils.distributed import get_rank
from . import MODEL_ZOO
from .modules.loss import BarlowTwinsLoss, VICRegLoss


class ViewpointBottleneck(nn.Module):

    def __init__(self, arch, mode='SSRL', criterion='BarlowTwinsLoss') -> None:
        super().__init__()
        self.mode = mode
        self.encoder_cls = MODEL_ZOO[arch]
        self.encoder = self.encoder_cls(3, 20, [1 for _ in range(8)], 0.02)
        self.fc = self.encoder.final
        self.encoder.final = nn.Identity()
        if mode == 'SSRL':
            self.criterion = BarlowTwinsLoss() if criterion == 'BarlowTwinsLoss' else VICRegLoss()
            self.split_transform = t.SplitCompose(
                sync_transform=[t.ToDevice(get_rank())],
                random_transform=[
                    t.RandomRotation(),
                    t.RandomTranslation(),
                    t.RandomScaling(),
                    t.ToSparseTensor(),
                ],
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
            self.split_transform = t.Compose([t.ToDevice(get_rank()), t.ToSparseTensor()])

    def train_step(self, input):
        if self.mode == 'SSRL':
            (tfield_a, tfield_sparse_a, _), (tfield_b, tfield_sparse_b, _), *_ = self.split_transform(*input)
            feats_a = self.encoder(tfield_sparse_a).slice(tfield_a)
            feats_b = self.encoder(tfield_sparse_b).slice(tfield_b)

            indices = p2.furthest_point_sampling(feats_a.C[:, 1:].unsqueeze(0).contiguous(), 1024).reshape(1024).long()

            ds_feats_a = torch.index_select(feats_a.F, 0, indices)
            ds_feats_b = torch.index_select(feats_b.F, 0, indices)

            return self.criterion(ds_feats_a, ds_feats_b)
        else:
            tfield, tfield_sparse, target, *_ = self.split_transform(*input)
            logits = self.fc(self.encoder(tfield_sparse)).slice(tfield).F
            return self.criterion(logits, target.long()), None

    def validate_step(self, input):
        getLogger(__name__).debug('before validate step')
        tfield, tfield_sparse, target, *_ = self.split_transform(*input)
        getLogger(__name__).debug('after transforms')
        logits = self.fc(self.encoder(tfield_sparse)).slice(tfield).F
        getLogger(__name__).debug('after forward')
        return torch.argmax(logits, dim=1), target

    def forward(self, input):
        if self.training:
            return self.train_step(input)
        else:
            return self.validate_step(input)
