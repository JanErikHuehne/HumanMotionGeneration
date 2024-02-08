import open3d as o3d
import os
import trimesh
import numpy as np

def visualize_mesh_sequence(mesh_files, offset_axis=2, offset_value=0.05):
    """
    Visualize a sequence of mesh frames in a single scene.

    Parameters:
    - mesh_files: List of paths to mesh files.
    - offset_axis: Axis along which to offset the meshes (0, 1, or 2 corresponding to x, y, z).
    - offset_value: The distance by which to offset each successive mesh.
    """
    scene = trimesh.Scene()  # Create an empty scene

    for i, mesh_file in enumerate(mesh_files):
        mesh = trimesh.load(mesh_file)  # Load the mesh
        translation_vector = np.zeros(3)
        translation_vector[offset_axis] = i * offset_value  # Calculate offset
        mesh.apply_translation(translation_vector)  # Apply the offset to the mesh
        scene.add_geometry(mesh)  # Add the mesh to the scene

    return scene

# Load the mesh
def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        return files
mesh_files = list_files(r'F:\ADL\CV\s2m_with_joint_position_loss\save\server_black2_0.5\13710_40_80\sample00_rep02_obj')
mesh = []
for f in mesh_files:
    object = os.path.join(r'F:\ADL\CV\s2m_with_joint_position_loss\save\server_black2_0.5\13710_40_80\sample00_rep02_obj', f)
    # mesh.append(trimesh.load(object))
    mesh.append(trimesh.load_mesh(object))

# # Visualize the sequence
# scene = visualize_mesh_sequence(mesh)
# scene.show()

# Visualize the mesh
trimesh.Scene([mesh]).show()