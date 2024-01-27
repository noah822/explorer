from __future__ import annotations
import numpy as np


from collections import namedtuple
from typing import List, Annotated, Literal, Union, Tuple

from explorer.agent.sensor import CameraSensor
from scipy.spatial.transform import Rotation as R

# type annotations
QUAT_SCALAR_LAST = np.ndarray     # quaternion in MAGNUM convention (w, x, y, z)
COORD_3D = Annotated[np.ndarray, Literal['N', 3]]
COORD_4D = Annotated[np.ndarray, Literal['N', 4]]



__all__ = [
    'CameraTransformer'
]

def _matmul_with_vec(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    perform (extended) matrix multiplication
    Args:
        A: (..., R, C)
        b: (..., C) 
    Returns:
        (..., C)
    '''
    return np.matmul(A, b[..., np.newaxis]).squeeze(-1)


class CameraTransformer:
    '''
    this class provides basic transformation between:
        1. world and camera coordinate 
        2. point cloud to depth image
    some of the codes are adapted from 
    https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/utils/depth.py

    '''
    CameraMatrix = namedtuple('CameraMatrix', ['intrinsic', 'extrinsic'])


    @staticmethod
    def from_instance(camera: CameraSensor, pos: np.ndarray, rot: QUAT_SCALAR_LAST, camera_centered=False) -> CameraTransformer:
        return CameraTransformer(camera.focal, camera.resolution, pos, rot, camera_centered)
        

    def __init__(self, f: float, resolution: List[int], pos: np.ndarray, rot: QUAT_SCALAR_LAST, camera_centered=False):
        '''
        Args:
            f: focal length in unit of pixel number
            resolution: size of image in unit of (#pixel, #pixel)
            pos: position of the camera relative to the world frame
            rot: quaternion denoting the rotating process of the camera
                 by default, it refers the rotation required to turn the world frame to camera's frame
            camera_centered:
                by default set to `False`, meaning that the rot parameter is relative to the world frame, 
                when set to `True`, rot is relative to the camer body frame
        '''
        self.h, self.w = resolution
        # principal point locates at the center of the image
        cx, cy = (self.h-1) / 2, (self.w-1) / 2 

        # shape of camera matrices: intrinsic: (3, 3), extrinsic: (4, 4)
        self.intrinsic = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ]) 
        self.extrinsic = np.zeros((4, 4))

        if not camera_centered:
           rot = R.from_quat(rot).inv().as_quat()
        rot_mat = R.from_quat(rot).as_matrix()

        self.extrinsic[:3, :3] = rot_mat
        self.extrinsic[:3, 3] = -rot_mat @ pos
        self.extrinsic[3, 3] = 1
        self.f = f
    
    def get_camera_matrics(self) -> CameraMatrix:
        '''
        Returns:
            (intrinsic, extrinsic) 
            we follow the convention here, extrinsic matrix computes the transformation from world frame to camera frame
        '''
        return self.CameraMatrix(self.intrinsic, self.extrinsic)
    
    def camera2world(self, coord: Union[COORD_3D, COORD_4D]) -> Union[COORD_3D, COORD_4D]:
        '''
        transform from camera coordinate to world coordinate, taking account of camera's position and orientation
        Args:
            1. coord: (..., 3 or 4), accepts coordinates in either homogenous coordinate or not
        Return:
            return transformed coordinates in world frame, 
            the format(whether in homo-coordinate or not) is determined by the input 
        '''
        B, D = coord.shape[:-1], coord.shape[-1]
        assert (D == 3 or D == 4), 'only accepts 3D or 4D coordinates'
        in_homo_coord = (D == 4)

        if not in_homo_coord:
            # not in homogenous coordinate
            # then convert it to homo coord by appending one at forth dimension
            coord = np.append(coord, np.ones((*B, 1)), axis=-1)

        world_coord = _matmul_with_vec(np.linalg.inv(self.extrinsic), coord)
        if not in_homo_coord:
            world_coord = world_coord[...,:3]

        return world_coord
    

    def repose(self, new_pos: np.ndarray, new_rot: QUAT_SCALAR_LAST):
        '''change the position and orientation of camera'''
        pass

    def world2camera(self, coord: COORD_3D) -> COORD_3D:
        '''
        transform from world coordinate to camera coordinate
        '''
        pass

    def world2image(self, coord: COORD_3D):
        pass

    def depth2point_cloud(self, depth_image: np.ndarray, down_sampling: int=1, heading_axis: str='-z') -> COORD_3D:
        '''
        convert a depth image to point cloud in camera's coordinate
        Args:
            depth_image: (..., H, W)
            down_sampling: sub-sampling a part of pixel from the depth image
            identifier: accepted format (-){x, y, z}
            specify the heading axis, the ordering of other two axes are determined under the convention of right-hand coord system
        Returns:
            point_cloud of shape (..., H, W, 3) in camera's frame
        '''
        # (0, 0) refers to the lower right corner of the image
        grid_x, grid_y = np.meshgrid(range(self.w), range(self.h-1, -1, -1))
        grid_x = grid_x[..., ::down_sampling, ::down_sampling]
        grid_y = grid_y[..., ::down_sampling, ::down_sampling]

        ready_to_transform = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1) # of shape (..., H, W, 3)
        ready_to_transform = ready_to_transform * depth_image[...,np.newaxis]

        point_cloud = _matmul_with_vec(np.linalg.inv(self.intrinsic), ready_to_transform)
        permuter, multiplier = CameraTransformer._parse_heading_axis_str(heading_axis)
        point_cloud = point_cloud[..., permuter] * multiplier

        return point_cloud 

    @staticmethod
    def _parse_heading_axis_str(identifier: str) -> Tuple[List[int], np.ndarray]:
        '''
        parse heading identifier string, ordering of other two axis is determined under the convention of right-hand coord system
        exhaust all possibilities 
        Args:
            identifier: accepted format (-){x, y, z}
        Returns:
            mutiplier & permuter
            processing procedure: raw_point_cloud -> permute -> multiply
        '''
        # TODO: better why of doing this
        multiplier, permuter = None, None
        identifier = identifier.lower()
        should_negate = identifier[0] == '-'
        
        if identifier[-1] == 'x':
            permuter = [2, 1, 0]
            multiplier = np.array([-1. if should_negate else 1., 1., 1.])
        elif identifier[-1] == 'y':
            permuter = [0, 2, 1]
            multiplier = np.array([1., -1. if should_negate else 1., 1.])
        elif identifier[-1] == 'z':
            permuter = [0, 1, 2]
            multiplier = np.array([1., 1., -1. if should_negate else 1.])
        else:
            raise ValueError('invalid identifier of heading axis, accepted format (-1){x, y, z}')
        
        return permuter, multiplier 
            

        
