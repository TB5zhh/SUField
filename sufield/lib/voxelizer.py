import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
from scipy.linalg import expm, norm


# Rotation matrix along axis with angle theta
# TODO verify this
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class VoxelizerBase():

    def __init__(self,
                 voxel_size=1,
                 clip_bound=None,
                 use_augmentation=False,
                 scale_augmentation_bound=None,
                 rotation_augmentation_bound=None,
                 translation_augmentation_ratio_bound=None,
                 ignore_label=255):
        """
        Args:
        - voxel_size: side length of a voxel, in meters?
        - clip_bound: boundary of the voxelizer. Points outside the bound will be deleted   
            expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).   
        - scale_augmentation_bound: None or (0.9, 1.1)  
        - rotation_augmentation_bound: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis. 
            Use random order of x, y, z to prevent bias.    
        - translation_augmentation_bound: ((-5, 5), (0, 0), (-10, 10))  
        - ignore_label: label assigned for ignore (not a training label).   
        """
        self.voxel_size = voxel_size
        self.clip_bound = clip_bound
        self.ignore_label = ignore_label

        # Augmentation
        self.use_augmentation = use_augmentation
        self.scale_augmentation_bound = scale_augmentation_bound
        self.rotation_augmentation_bound = rotation_augmentation_bound
        self.translation_augmentation_ratio_bound = translation_augmentation_ratio_bound

    def get_transformation_matrix(self):
        '''
        Get voxelization (randomly scaled) matrix and random rotation matrix
        '''
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)
        ''' 
        A. Random rotation
            In random order of x,y,z 
            rotate the point cloud with random angle in the range
            if specified rotate range in `rotation_augmentation_bound`
        '''
        rot_mat = np.eye(3)
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(*rot_bound)
                rot_mats.append(M(axis, theta))
            # In random order
            np.random.shuffle(rot_mats)
            rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            rotation_matrix[:3, :3] = rot_mat
        '''
        B. Random Scaling
            First calculate a fix scale ratio
            Then choose a random coefficient in the range
            and get the actual scale ratio
        '''
        scale = 1 / self.voxel_size
        if self.use_augmentation and self.scale_augmentation_bound is not None:
            scale *= np.random.uniform(*self.scale_augmentation_bound)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)

        return voxelization_matrix, rotation_matrix

    def clip(self, coords, center=None):
        '''
        Args:
        - coords: the coordinates of a pointcloud
        - center: the center of clipping. If not set, then treat the middle place of point cloud as center
        - TODO update this trans_aug_ratio: the ratio of translation. The actual translation will be `ratio * bound_box_size`
            the bound_box_size will be the rangeS of coordinates

        Return:
        - the indices of vertices inside clip limits
        '''
        # TODO check
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)

        bound_size = bound_max - bound_min

        # Choose the center of the pointcloud
        if center is None:
            center = bound_min + bound_size * 0.5

        trans_aug_ratio = np.zeros(3)
        if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
            for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

        trans = np.multiply(trans_aug_ratio, bound_size)
        center += trans

        # Clip points outside the limit
        if self.clip_bound[0] is not None:
            clip_indx_0 = (coords[:, 0] >= (self.clip_bound[0][0] + center[0])) & \
                (coords[:, 0] < (self.clip_bound[0][1] + center[0]))
        else:
            clip_indx_0 = np.ones((coords.shape[0]), dtype=bool)
        if self.clip_bound[1] is not None:
            clip_indx_1 = (coords[:, 1] >= (self.clip_bound[1][0] + center[1])) & \
                (coords[:, 1] < (self.clip_bound[1][1] + center[1]))
        else:
            clip_indx_1 = np.ones((coords.shape[0]), dtype=bool)
        if self.clip_bound[2] is not None:
            clip_indx_2 = (coords[:, 2] >= (self.clip_bound[2][0] + center[2])) & \
                (coords[:, 2] < (self.clip_bound[2][1] + center[2]))
        else:
            clip_indx_2 = np.ones((coords.shape[0]), dtype=bool)

        clip_inds = clip_indx_0 & clip_indx_1 & clip_indx_2
        return clip_inds

    def prevoxelize_transform(self, coords, feats, labels, center=None):
        assert coords.shape[1] == 3, 'Coordinates have dimensions other than 3'
        assert coords.shape[0] == feats.shape[0], 'different batch size for coords and feats'
        '''
        Clip the pointcloud if given a clip bound
        including random tranlation of center
        '''
        if self.clip_bound is not None:
            clip_inds = self.clip(coords, center)
            coords, feats = coords[clip_inds], feats[clip_inds]
            if labels is not None:
                labels = labels[clip_inds]
        '''
        Get scale (mainly for voxelization, random scale included) matrix and rotation matrix
        And apply them to the coords
        '''
        M_v, M_r = self.get_transformation_matrix()
        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords_aug = np.floor(homo_coords @ (M_r @ M_v).T[:, :3])
        '''
        Translate the point cloud so that all points have positive coordinates
        '''
        min_coords = coords_aug.min(0)
        coords_aug = np.floor(coords_aug - min_coords)

        M_t = np.eye(4)
        M_t[:3, -1] = -min_coords
        rigid_transformation = M_t @ M_r @ M_v

        return coords_aug, feats, labels, rigid_transformation

    def voxelize(self):
        raise NotImplementedError


class Voxelizer(VoxelizerBase):

    def voxelize(self, coords1, coords2, feats1, labels1, feats2, labels2, center=None):
        coords1_aug, feats1_aug, labels1_aug, rigid_transformation1 = self.prevoxelize_transform(coords1, feats1, labels1, center)
        coords2_aug, feats2_aug, labels2_aug, rigid_transformation2 = self.prevoxelize_transform(coords2, feats2, labels2, center)

        coords1_aug, feats1_aug, labels1_aug, indices_map = ME.utils.sparse_quantize(coords1_aug,
                                                                                     feats1_aug,
                                                                                     labels=labels1_aug,
                                                                                     ignore_label=self.ignore_label,
                                                                                     return_index=True)
        coords2_aug, feats2_aug, labels2_aug = coords2_aug[indices_map], feats2_aug[indices_map], labels2_aug[indices_map]

        return coords1_aug, feats1_aug, labels1_aug, coords2_aug, feats2_aug, labels2_aug, (rigid_transformation1, rigid_transformation2, indices_map)
        # try:
        #     coords1_aug = coords1_aug.numpy()
        # except:
        #     pass


class TestVoxelizer(VoxelizerBase):

    def voxelize(self, coords, feats, labels, center=None):
        coords_aug, feats_aug, labels_aug, rigid_transformation = self.prevoxelize_transform(coords, feats, labels, center)

        coords_aug, feats_aug, labels_aug, indices = ME.utils.sparse_quantize(coords_aug,
                                                                              feats_aug,
                                                                              labels=labels_aug,
                                                                              ignore_label=self.ignore_label,
                                                                              return_index=True)

        return coords_aug, feats_aug, labels_aug, (rigid_transformation, indices)


def test():
    N = 16
    coords = np.random.rand(N, 3) * 10
    feats = np.random.rand(N, 4)
    labels = np.floor(np.random.rand(N) * 3)
    coords[:3] = 0
    labels[:3] = 2
    voxelizer = TestVoxelizer(
        voxel_size=0.02,
        clip_bound=(None, None, None),
        use_augmentation=True,
        scale_augmentation_bound=(0.9, 1.1),
        rotation_augmentation_bound=((np.pi / 2, np.pi / 2), None, None),
    )
    print(coords.shape)
    print(feats.shape)
    print(labels.shape)
    coords_aug, feats_aug, labels_aug, (transformation, indices) = voxelizer.voxelize(coords, feats, labels)
    print(coords)
    print(coords_aug)
    print("=======")
    print("=======")
    print(feats)
    print(feats_aug)
    print("=======")
    print("=======")
    print(labels)
    print(labels_aug)
    print("=======")
    print("=======")
    print(transformation)
    print(indices)


if __name__ == '__main__':
    test()
