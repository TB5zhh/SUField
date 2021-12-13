import numpy as np
from sufield.lib.voxelizer import Voxelizer, TestVoxelizer

class TestVoxelizer():
    def test_testvoxelizer_create(self):
        N = 16575
        coords = np.random.rand(N, 3) * 10
        feats = np.random.rand(N, 4)
        labels = np.floor(np.random.rand(N) * 3)
        coords[:3] = 0
        labels[:3] = 2
        voxelizer = TestVoxelizer()
        print(voxelizer.voxelize(coords, feats, labels))
        assert True