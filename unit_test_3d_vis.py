import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def create_ellipsoid_mesh(scales, density=100):
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

# Create a list to store ellipsoid meshes
ellipsoid_meshes = []

# Generate ellipsoid meshes with colors, positions, and rotations
for params in ellipsoid_parameters:
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
o3d.io.write_triangle_mesh("multi_ellipsoids_with_transformations.ply", combined_mesh)
