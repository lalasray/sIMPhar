import numpy as np
import os

def load_obj(filename):
    vertices = []
    with open(filename, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
    return np.array(vertices)

def calculate_spatial_distance(vertices, start_index, end_index):
    start_vert = vertices[start_index]
    end_vert = vertices[end_index]
    spatial_distance = np.linalg.norm(end_vert - start_vert)
    return spatial_distance

def save_to_txt(filename, data):
    with open(filename, 'w') as f:
        f.write(str(data))

start_index = 4824 #smplx
end_index = 7560 #smplx


obj_directory = r"C:\Users\lalas\Desktop\sIMphar\S1_1_PXL_20240405_121105988\mesh"

for filename in os.listdir(obj_directory):
    if filename.endswith(".obj"):
        mesh_file_path = os.path.join(obj_directory, filename)
        vertices = load_obj(mesh_file_path)
        spatial_distance = calculate_spatial_distance(vertices, start_index, end_index)
        output_file = obj_directory+"/"+filename.replace(".obj", ".txt")
        save_to_txt(output_file, spatial_distance)
        print("Spatial distance saved to:", output_file)
