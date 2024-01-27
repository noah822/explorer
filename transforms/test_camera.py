import numpy as np

from scipy.spatial.transform import Rotation


from camera import CameraTransformer
from voxel import splat_feature


def _euler_angle2quat(angles, body_centered=True):
    '''
    transform from [x, y, z] rotation angle to quat
    x, y, z are in unit of degree
    '''
    seq = 'XYZ' if body_centered else 'xyz'
    return Rotation.from_euler(seq, angles, degrees=True).as_quat()
    

class TestCamera:
    '''
    handy website to check common angular transformations:
    https://www.andre-gaschler.com/rotationconverter
    '''

    def test_camera2world(self):
        camera1 = CameraTransformer(
            10, (100, 100),
            np.array([3, 2, 0]),
            _euler_angle2quat([0, 0, 90])
        ) 
        camera_frame1 = np.array([[1, 1, 1]])
        world_frame1 = camera1.camera2world(camera_frame1)   
        assert np.allclose(world_frame1, np.array([[2, 3, 1]]))

        camera2 = CameraTransformer(
            10, (100, 100),
            np.array([3, 2, 0]),
            _euler_angle2quat([90, 30, 0])
        ) 
        camera_frame2 = np.array([[np.sqrt(3), 10, 1]])
        world_frame2 = camera2.camera2world(camera_frame2)
        assert np.allclose(world_frame2, np.array([[5, 2, 10]]))
    

    def test_world2image(self):
        pass

    def test_depth2point_cloud(self):
        camera = CameraTransformer(
            1, [2, 2],
            np.array([0, 0, 0]),
            _euler_angle2quat([0, 0, 0])
        )
        depth_image = np.array([[1, 0.8], [1.2, 1]])
        point_cloud = camera.depth2point_cloud(depth_image, heading_axis='z')
        ans = np.array([
            [[-0.5, 0.5, 1],[0.4, 0.4, 0.8]],
            [[-0.6, -0.6, 1.2],[0.5, -0.5, 1]],
        ])
        assert np.allclose(point_cloud, ans)

        


def _opengl_normalize(coords, l, r):
    '''
    convention widely adopted in OpenGL convention
    normalize coordinates into [-1, 1] according to the range of each dimension
    Args:
        coords: (..., D)
        l/r: (D, )
    '''
    return (coords - (l+r)/2)*2 / (l+r)
     


def test_splat_feature():
    B = 1; numPt = 1; F = 1; D = 2
    H = W = 3
    l = np.array(0); r = np.array(2)
    features = np.array([[[1]]])
    coords = np.array([[[1.5, 1.5]]])
    normed_coords = _opengl_normalize(coords, l, r)
    grids = np.zeros((B, H, W, F))
    splat_feature(features, normed_coords, grids)
    region = grids[0,1:,1:,:]
    assert np.allclose(
        region,
        np.array([[[0.25], [0.25]], [[0.25], [0.25]]])
    )

    coords = np.array([[[0.5, 0.5], [1.5, 1.5]]])
    normed_coords = _opengl_normalize(coords, l, r)
    grids = np.zeros((B, H, W, F))
    features = np.array([[[1], [1]]])
    splat_feature(features, normed_coords, grids)
    ans = np.array([[0.25, 0.25, 0], [0.25, 0.5, 0.25], [0, 0.25, 0.25]])
    assert np.allclose(grids.squeeze(), ans)


