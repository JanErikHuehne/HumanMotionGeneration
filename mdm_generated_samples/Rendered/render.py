import trimesh
import pyrender
import os
# Path to directory where .obj files 
directory = "/media/jan/SSD Spiele/ADLCV/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_the_person_dances/sample00_rep00_obj"
files = os.listdir(directory)
scene = pyrender.Scene()

v = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True, record=True)
node = None

for i,f in enumerate(files):
    print(i)
    v.render_lock.acquire()
    if node:
        scene.remove_node(node)
    mesh = trimesh.load(os.path.join(directory, f))
    mesh = pyrender.Mesh.from_trimesh(mesh)
    node = pyrender.Node(mesh=mesh)
    scene.add_node(node)
    v.render_lock.release()
print("Finished")
v.close_external()
# Output path for the gif
v.save_gif("test4.gif")

