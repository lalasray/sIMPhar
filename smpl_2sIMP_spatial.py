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

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".obj"):
                mesh_file_path = os.path.join(root, filename)
                vertices = load_obj(mesh_file_path)
                if len(vertices) < max(start_index, end_index) + 1:
                    print("Not enough vertices in:", filename)
                    continue
                spatial_distance = calculate_spatial_distance(vertices, start_index, end_index)
                output_file = os.path.join(root, filename.replace(".obj", ".txt"))
                save_to_txt(output_file, spatial_distance)
                print("Spatial distance saved to:", output_file)

start_index = 4824  # smplx
end_index = 7560  # smplx
obj_directory = r"C:\Users\lalas\Desktop\sIMphar"

# Check if directory exists
if not os.path.exists(obj_directory):
    print("Directory does not exist:", obj_directory)
    exit()

process_directory(obj_directory)
