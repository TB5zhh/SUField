from sufield.config import VALID_CLASS_IDS
from sufield.lib.dataset import VoxelizedDataset, VoxelizedTestDataset

class TestDatsetVariant:
    def test_train_variant(self):
        class Derived(VoxelizedDataset):
            IGNORE_LABELS =  tuple(set(range(41)) - set(VALID_CLASS_IDS))
        dataset = Derived('')
        assert dataset.VARIANT == 'train'
    def test_test_variant(self):
        class Derived(VoxelizedDataset):
            IGNORE_LABELS =  tuple(set(range(41)) - set(VALID_CLASS_IDS))
        dataset = Derived('')
        assert dataset.VARIANT == 'test'