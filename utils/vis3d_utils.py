import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


def create_ellipsoid_mesh(scales, density=20):
    # Generate points on a sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=density)
    points = np.asarray(sphere.vertices)

    # Scale points along each axis to form an ellipsoid
    points[:, 0] *= scales[0]
    points[:, 1] *= scales[1]
    points[:, 2] *= scales[2]

    # Create triangle mesh from points
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = sphere.triangles

    return mesh


def view_3d(means_3d, scales, quats, save_path, colors=None):
    if colors is None:
        colors = generate_colors(len(means_3d))

    ps_info = []

    for i in range(len(means_3d)):
        ps_info.append({
            'scales': scales[i].tolist(),
            'color': colors[i].tolist(),
            'position': means_3d[i].tolist(),
            'quat': quats[i].tolist(),
        })
    
    save_3d_meshes(ps_info, save_path)


def save_3d_meshes(ps_info, save_path):
    # Create a list to store ellipsoid meshes
    ellipsoid_meshes = []

    # Generate ellipsoid meshes with colors, positions, and rotations
    for params in ps_info:
        scales = params['scales']
        color = params['color']
        position = params['position']
        quat = params['quat']
        
        # Create ellipsoid mesh
        ellipsoid_mesh = create_ellipsoid_mesh(scales)
        ellipsoid_mesh.paint_uniform_color(color)
        
        # Apply translation to the ellipsoid
        translation = np.identity(4)
        translation[:3, 3] = position
        
        # Convert quaternion to rotation matrix
        rotation_matrix = R.from_quat(quat).as_matrix()
        
        # Combine translation and rotation
        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix = np.dot(translation, transformation_matrix)
        
        # convert opengl to opencv coordinate system for visualization
        transformation_matrix = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) @ transformation_matrix

        ellipsoid_mesh.transform(transformation_matrix)
        
        ellipsoid_meshes.append(ellipsoid_mesh)

    # Combine all ellipsoid meshes into a single mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    for ellipsoid_mesh in ellipsoid_meshes:
        combined_mesh += ellipsoid_mesh

    # Save to PLY file
    o3d.io.write_triangle_mesh(save_path, combined_mesh)



def generate_colors(num):
    values = np.arange(num) / num
    colors = [jet_color_map(value) for value in values]

    return np.array(colors).reshape(-1, 3)


def jet_color_map(value):
    value = max(0.0, min(1.0, value))  # Clip value between 0 and 1
    r = int(max(0, min(255, 255 * min(4 * value - 1.5, -4 * value + 4.5))))
    g = int(max(0, min(255, 255 * min(4 * value - 0.5, -4 * value + 3.5))))
    b = int(max(0, min(255, 255 * min(4 * value + 0.5, -4 * value + 2.5))))
    return [r / 255, g / 255, b / 255]


if __name__ == '__main__':
    # Define parameters for multiple ellipsoids
    ellipsoid_parameters = [
        {'scales': [0.2, 0.15, 0.1], 'color': [0.8, 0.2, 0.2], 'position': [0, 0, 0], 'quat': [1, 0, 0, 0]},  # Red
        {'scales': [0.2, 0.15, 0.1], 'color': [0.8, 0.2, 0.2], 'position': [0, 0.5, 0], 'quat': [1, 0, 0, 0]},  # Red
        {'scales': [0.15, 0.1, 0.05], 'color': [0.2, 0.8, 0.2], 'position': [0, -0.5, 0], 'quat': [1, 0, 0, 0]},  # Green
        {'scales': [0.2, 0.15, 0.1], 'color': [0.8, 0.2, 0.2], 'position': [0.5, 0, -1], 'quat': [1, 0, 0, 0]},  # Red
        {'scales': [0.15, 0.1, 0.05], 'color': [0.2, 0.8, 0.2], 'position': [-0.5, 0, -1], 'quat': [1, 0, 0, 0]},  # Green
        {'scales': [0.15, 0.1, 0.05], 'color': [0.2, 0.8, 0.2], 'position': [0, 0, 1], 'quat': [0.9238795, 0, 0, 0.3826834]},  # Green
        {'scales': [0.1, 0.2, 0.075], 'color': [0.2, 0.2, 0.8], 'position': [0, 0, -1], 'quat': [0.7071068, 0, 0, 0.7071068]}  # Blue
    ]

    save_3d_meshes(ps_info=ellipsoid_parameters, save_path='multi_ellipsoids_with_transformations.ply')