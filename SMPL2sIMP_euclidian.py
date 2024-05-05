import bpy
import os
import math

# Function to calculate the spatial distance between two vertices
def calculate_spatial_distance(mesh, start_index, end_index, path_indices):
    verts = mesh.vertices
    spatial_distance = 0
    for i in range(len(path_indices) - 1):
        start_vert = verts[path_indices[i]].co
        end_vert = verts[path_indices[i + 1]].co
        spatial_distance += (end_vert - start_vert).length
    return spatial_distance

# Function to calculate the Euclidean distance between two vertices
def calculate_euclidean_distance(mesh, start_index, end_index):
    verts = mesh.vertices
    start_vert = verts[start_index].co
    end_vert = verts[end_index].co
    return (end_vert - start_vert).length

# Function to find the shortest path between two vertices
def find_shortest_path(mesh, start_index, end_index):
    verts = mesh.vertices
    edges = mesh.edges
    loops = mesh.loops
    visited = [False] * len(verts)
    visited[start_index] = True
    queue = [[start_index]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == end_index:
            return path
        for loop_index in [l.index for l in loops if l.vertex_index == node]:
            loop = loops[loop_index]
            edge_index = loop.edge_index
            edge = edges[edge_index]
            next_vert_index = edge.vertices[0] if edge.vertices[1] == node else edge.vertices[1]
            if not visited[next_vert_index]:
                visited[next_vert_index] = True
                new_path = list(path)
                new_path.append(next_vert_index)
                queue.append(new_path)
    return None

# Function to save data to a text file
def save_to_txt(mesh_name, filename, *data):
    with open(filename, 'w') as f:
        f.write(','.join(map(str, data)))

# Directory containing OBJ files
obj_directory = "/home/lala/Desktop/sIMPhar/"

# Iterate through all files in the directory
for filename in os.listdir(obj_directory):
    if filename.endswith(".obj"):
        mesh_file_path = os.path.join(obj_directory, filename)
        bpy.ops.import_scene.obj(filepath=mesh_file_path)
        
        # Iterate through all mesh objects in the scene
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                mesh = obj.data
                mesh_name = os.path.splitext(os.path.basename(mesh_file_path))[0]
                print(mesh_name)
                start_index = 4824
                end_index = 7560
                path_indices = find_shortest_path(mesh, start_index, end_index)
                if path_indices:
                    spatial_distance = calculate_spatial_distance(mesh, start_index, end_index, path_indices)
                    euclidean_distance = calculate_euclidean_distance(mesh, start_index, end_index)
                    num_edges = len(path_indices) - 1
                    save_to_txt(mesh_name, os.path.join(obj_directory, "{}.txt".format(mesh_name)), spatial_distance, euclidean_distance)
                    print("Mesh:", mesh_name)
                    print("Spatial distance along the path:", spatial_distance)
                    print("Euclidean distance between the two vertices:", euclidean_distance)
                    print("Number of edges along the path:", num_edges)
                    print("Connected vertices:", path_indices)
                else:
                    print("No path found between the vertices for mesh:", mesh_name)
            else:
                print("Object", obj.name, "is not a mesh.")
        
        # Delete the imported mesh object
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
                bpy.ops.object.delete()
    else:
        print("Skipping non-OBJ file:", filename)
