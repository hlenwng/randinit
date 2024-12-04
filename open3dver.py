import cv2
import apriltag
import json
import numpy as np
import open3d as o3d
import threading
import time
import random
import csv
import os
from datetime import datetime
from scipy.spatial import cKDTree

tag_size = 0.06  # 6cm square

# Load CAD mesh using Open3D
mesh_name = 'leg_cad'
cad_mesh = o3d.io.read_triangle_mesh(f'/Users/helenwang/helen_lars/meshes/{mesh_name}.ply')
cad_mesh = cad_mesh.subdivide_midpoint(number_of_iterations=1)  # Increase iterations for finer mesh

#cad_mesh = cad_mesh.simplify_quadric_decimation(target_number_of_triangles=500)  # Simplify the mesh for faster rendering
cad_mesh.compute_vertex_normals()

csv_filename = 'mesh_positions.csv'
if not os.path.isfile(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mesh_name', 'x', 'y', 'z', 'time']) 

# Store IRL coordinates of tag corners
obj_points = np.array([
    [-tag_size / 2, -tag_size / 2, 0],
    [tag_size / 2, -tag_size / 2, 0],
    [tag_size / 2, tag_size / 2, 0],
    [-tag_size / 2, tag_size / 2, 0]
], dtype=np.float32)

# Load camera intrinsics from a given JSON file
try:
    with open('camera_intrinsics.json') as f:
        intrinsics = json.load(f)
except FileNotFoundError:
    print("Error: camera_intrinsics.json file not found.")
    exit(1)
if 'camera_matrix' not in intrinsics or 'distortion_coefficients' not in intrinsics:
    print("Error: Missing camera intrinsics in JSON.")
    exit(1)

camera_matrix = np.array(intrinsics['camera_matrix'], dtype=np.float32)
dist_coeffs = np.array(intrinsics['distortion_coefficients'], dtype=np.float32)

# Initialize AprilTag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Stored variables
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Initial position of box
final_box_coords = None  # Final box coordinates after button click
saved_box_coords = None  # To store the blue box coordinates after pressing 'c'
draw_box_mode = False  # Toggle to enable drawing mode

cad_mesh_vertices = None  # To store projected vertices of the CAD mesh
selected_face_index = None  # To store the selected face index
user_choice = None
matrix_rotation = None # To store the rotation matrix based on selected face index
adjusted_scaling = 1.25  # To fine-tune the scaling of displayed mesh
key_pressed = None  # To store the key pressed by the user

shift_height = 0  # Height shift for mesh alignment to z=0
translation_step = 0.1  # Translation step for mesh movement

'''
Functions for CAD mesh manipulation and visualization
'''

def draw_box(event, x_mouse, y_mouse, flags, param):
    global ix, iy, drawing, final_box_coords
    
    if not draw_box_mode: 
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_mouse, y_mouse  # Set the initial position

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            final_box_coords = [(ix, iy), (x_mouse, y_mouse)]
            print(f"Drawing box from {final_box_coords[0]} to {final_box_coords[1]}")

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def get_user_input():
    global user_choice
    while True:
        print("\nChoose a face direction to color (only one at a time):")
        print("1. Front, 2. Back, 3. Left, 4. Right, 5. Top, 6. Bottom, 0. Exit")
        try:
            choice = int(input("Enter the direction number: "))
            user_choice = choice
            if choice == 0:
                break
        except ValueError:
            print("Please enter a valid number.")

def align_face_to_bottom(mesh, face_index, scaling):
    # Compute normals for the mesh
    mesh.compute_vertex_normals()

    # Get the vertices of the specified face
    triangle = np.asarray(mesh.triangles)[face_index]
    vertices = np.asarray(mesh.vertices)[triangle]

    # Calculate the normal of the face
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    normal /= np.linalg.norm(normal)

    # Target direction is the downward z-axis
    target_direction = np.array([0, 0, -1])  # Flip for down direction
    normal = -normal  # Flip the normal to point in the correct direction

    # Create the rotation matrix to align the face normal with the target direction
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_two_vectors(normal, target_direction)

    mesh.rotate(rotation_matrix, center=mesh.get_center())
    mesh.scale(scaling, center=mesh.get_center())

    return mesh

def shift_mesh_z(cad_mesh):

    cad_mesh.compute_vertex_normals()

    # Calculate height of the object
    vertices = np.asarray(cad_mesh.vertices)
    height = vertices[:, 2].max() - vertices[:, 2].min()

    return height

def display_cad_mesh(mesh):
    global user_choice, selected_face_index, matrix_rotation
    
    # Assign faces to each of the six primary directions
    directions = assign_faces_to_directions(mesh)
    reset_mesh_color(mesh)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="CAD Mesh", width=800, height=600)
    vis.add_geometry(mesh)

    input_thread = threading.Thread(target=get_user_input)
    input_thread.start()

    # Visualization and input processing loop
    while True:
        # Check if there's a new user input
        if user_choice is not None:
            choice = user_choice
            user_choice = None  # Reset choice after processing

            if choice == 0:
                break

            # Reset mesh color before applying new selection
            reset_mesh_color(mesh)
            selected_color = [1.0, 0.0, 0.0]  

            # Map user choice to direction and color the face
            if choice == 1:
                color_faces(mesh, directions["front"], selected_color)
                print("Colored the front faces.")
            elif choice == 2:
                color_faces(mesh, directions["back"], selected_color)
                print("Colored the back faces.")
            elif choice == 3:
                color_faces(mesh, directions["left"], selected_color)
                print("Colored the left faces.")
            elif choice == 4:
                color_faces(mesh, directions["right"], selected_color)
                print("Colored the right faces.")
            elif choice == 5:
                color_faces(mesh, directions["top"], selected_color)
                print("Colored the top faces.")
            elif choice == 6:
                color_faces(mesh, directions["bottom"], selected_color)
                print("Colored the bottom faces.")
            else:
                print("Invalid choice. Please select a number between 0 and 6.")

            # Update mesh visualization with new colors
            vis.update_geometry(mesh)
        
        # Poll events and update the renderer
        vis.poll_events()
        vis.update_renderer()
        
        time.sleep(0.01)

    vis.destroy_window()
    input_thread.join()  # Ensure input thread is cleaned up

def assign_faces_to_directions(mesh):
    # Dictionary to store the faces in each direction
    directions = {
        "front": [],
        "back": [],
        "left": [],
        "right": [],
        "top": [],
        "bottom": []
    }
    
    # Calculate the normal of each face to determine its orientation
    mesh.compute_triangle_normals()
    triangle_normals = np.asarray(mesh.triangle_normals)

    # Classify each face based on its normal
    for idx, normal in enumerate(triangle_normals):
        if np.dot(normal, [0, 0, 1]) > 0.9:        # Z+ direction -> front
            directions["front"].append(idx)
        elif np.dot(normal, [0, 0, -1]) > 0.9:     # Z- direction -> back
            directions["back"].append(idx)
        elif np.dot(normal, [1, 0, 0]) > 0.9:      # X+ direction -> right
            directions["right"].append(idx)
        elif np.dot(normal, [-1, 0, 0]) > 0.9:     # X- direction -> left
            directions["left"].append(idx)
        elif np.dot(normal, [0, 1, 0]) > 0.9:      # Y+ direction -> top
            directions["top"].append(idx)
        elif np.dot(normal, [0, -1, 0]) > 0.9:     # Y- direction -> bottom
            directions["bottom"].append(idx)

    return directions

def reset_mesh_color(mesh, default_color=[0.8, 0.8, 0.8]):
    # Set the entire mesh to the default color
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(default_color, (len(mesh.vertices), 1)))

def color_faces(mesh, faces, color):
    # Color the specified faces
    vertex_colors = np.asarray(mesh.vertex_colors)
    for face_idx in faces:
        for vertex_idx in mesh.triangles[face_idx]:
            vertex_colors[vertex_idx] = color
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

def update_mesh_location(mesh, new_location):
    # Translate mesh to a new location
    current_center = np.mean(np.asarray(mesh.vertices), axis=0)
    translation = new_location - current_center
    translation[2] = 0  # z-axis location is fixed

    mesh.translate(translation)

def project_and_print_center(mesh, camera_matrix, dist_coeffs):
    # Get the vertices of the CAD mesh
    vertices = np.asarray(mesh.vertices)
    
    # Compute the center of the mesh (mean of all vertices)
    center_3d = vertices.mean(axis=0)
    #print(f"Center of CAD Mesh (3D): {center_3d}")

    # Project the 3D center to 2D
    projected_center, _ = cv2.projectPoints(
        np.array([center_3d]),  # The 3D center point
        np.zeros(3),  # Rotation vector (already applied via `matrix_rotation`)
        np.zeros(3),  # Translation vector (apply if needed)
        camera_matrix,  # Camera intrinsic matrix
        dist_coeffs  # Distortion coefficients
    )

    # Reshape and convert to integer for display
    projected_center = projected_center.reshape(-1, 2).astype(int)
    #print(f"Center of CAD Mesh (2D): {projected_center[0]}")
    return projected_center[0]

def convert_to_3d(box_coords, rvec, tvec, camera_matrix, dist_coeffs):
    """Convert the 2D bounding box coordinates to 3D world coordinates."""
    
    # Define the 3D object points (bounding box in the world frame)
    object_points_3d = np.array([
        [-tag_size / 2, -tag_size / 2, 0],  # Bottom-left corner
        [tag_size / 2, -tag_size / 2, 0],   # Bottom-right corner
        [tag_size / 2, tag_size / 2, 0],    # Top-right corner
        [-tag_size / 2, tag_size / 2, 0]    # Top-left corner
    ], dtype=np.float32)
    
    # Define the 2D points of the bounding box in image space
    box_points_2d = np.array([
        [box_coords[0][0], box_coords[0][1]],  # Top-left
        [box_coords[1][0], box_coords[0][1]],  # Top-right
        [box_coords[1][0], box_coords[1][1]],  # Bottom-right
        [box_coords[0][0], box_coords[1][1]]   # Bottom-left
    ], dtype=np.float32)
    
    # Use solvePnP to get the rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(object_points_3d, box_points_2d, camera_matrix, dist_coeffs)
    
    if not success:
        raise ValueError("solvePnP failed to find a valid solution.")
    
    # Now, project the object points from 3D to 2D space
    projected_points_2d, _ = cv2.projectPoints(object_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    
    # Convert the projected points from (4, 1, 2) to (4, 2)
    projected_points_2d = projected_points_2d.reshape(-1, 2)
    
    # For each 2D point, map back to 3D space using the depth (z-value)
    # In this case, we'll use the z-value of the tag as the depth (it’s in meters)
    projected_points_3d = []
    for point_2d in projected_points_2d:
        # Here, you may need to adjust based on the actual z-depth or tag size
        # For simplicity, we'll assume z=0 (flat plane), or you may use tvec[2]
        projected_points_3d.append([point_2d[0], point_2d[1], 0])
    
    # Convert the list to a numpy array for better handling
    projected_points_3d = np.array(projected_points_3d)
    
    return projected_points_3d

def save_position_to_csv(position, filename, mesh_name):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([mesh_name, *position, current_time])

def check_size(cad_mesh, box_width, box_height):
    cad_mesh_vertices = np.asarray(cad_mesh.vertices)
    mesh_x = np.max(cad_mesh_vertices[:, 0]) - np.min(cad_mesh_vertices[:, 0])
    mesh_y = np.max(cad_mesh_vertices[:, 1]) - np.min(cad_mesh_vertices[:, 1])

    print(f"Mesh size: {mesh_x} x {mesh_y}")
    print(f"Box size: {box_width} x {box_height}")

    if abs(box_width) < mesh_x or abs(box_height) < mesh_y:
        return False
    
    return True


# Create window
cv2.namedWindow('April Tag Detection', cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback('April Tag Detection', draw_box)


# Open camera
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    key = cv2.waitKey(1) & 0xFF

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for detection in detections:
        # Draw bounding box around detected AprilTag
        corners = detection.corners.astype(int)  # Convert to integers for drawing
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=4)

        # Get rotation vector and translation vector
        success, rvec, tvec = cv2.solvePnP(obj_points, detection.corners, camera_matrix, dist_coeffs)

        if success and saved_box_coords is not None:

            '''
            Draw user-defined box that defines the working plane
            '''

            projected_center = project_and_print_center(cad_mesh, camera_matrix, dist_coeffs)

            # Convert saved box (2D) into a 3D box relative to the tag (pixel units)
            box_width = (saved_box_coords[1][0] - saved_box_coords[0][0]) / 1000.0  # Scale to meters
            box_height = (saved_box_coords[1][1] - saved_box_coords[0][1]) / 1000.0  # Scale to meters

            box_3d = np.array([
                [0, 0, 0],
                [box_width, 0, 0],
                [box_width, box_height, 0],
                [0, box_height, 0]
            ], dtype=np.float32)

            # Transform the 3D box corners using rvec and tvec
            box_3d_transformed, _ = cv2.projectPoints(box_3d, rvec, tvec, camera_matrix, dist_coeffs)

            # Draw the transformed box (blue)
            box_3d_transformed = box_3d_transformed.squeeze().astype(int)
            cv2.polylines(frame, [box_3d_transformed], isClosed=True, color=(255, 0, 0), thickness=2)
            
            #from ipdb import set_trace; set_trace()

            '''
            Project CAD mesh onto plane
            '''

            face_index = None  # To store the selected face index

            # Assuming 'face_index' is set from the face selection process
            if face_index is not None:
                cad_mesh = align_face_to_bottom(cad_mesh, face_index, adjusted_scaling)
                print(f"Aligned face {face_index} to bottom.")

            # Calculate CAD mesh scaling based on Tag size
            cad_bounds = np.asarray(cad_mesh.get_axis_aligned_bounding_box().get_extent())  # Get bounding box of CAD mesh
            cad_size = np.linalg.norm(cad_bounds)  # Calculate the size of the CAD mesh
            center = cad_mesh.get_center()
            
            if key_pressed is not None:
                cad_mesh.scale(adjusted_scaling, center)
                key_pressed = None


            # Prepare to project the CAD mesh
            if cad_mesh_vertices is not None:

                # Project the transformed CAD mesh vertices onto the 2D frame
                projected_vertices, _ = cv2.projectPoints(cad_mesh_vertices, rvec, tvec, camera_matrix, dist_coeffs)
                projected_vertices = projected_vertices.reshape(-1, 2).astype(int)

                # Draw the edges of the CAD mesh on the frame
                for i, vertex in enumerate(projected_vertices):
                    next_vertex = projected_vertices[(i + 1) % len(projected_vertices)]  # Wrap around to create a closed loop
                    # Ensure each vertex only contains two elements and convert them to integers
                    cv2.line(frame, tuple(map(int, vertex[:2])), tuple(map(int, next_vertex[:2])), (0, 0, 255), thickness=2)  # Draw red edges

            # Display the final frame
            cv2.imshow('April Tag Detection', frame)

    # Display the currently drawn rectangle if coordinates are set
    if final_box_coords is not None and final_box_coords[0] != final_box_coords[1]:
        cv2.rectangle(frame, final_box_coords[0], final_box_coords[1], (0, 255, 0), 2)

    # Show the frame with AprilTags and the drawn box
    cv2.imshow('April Tag Detection', frame)

    '''
    Check for key presses that trigger actions
    Eventually, convert these to buttons within the GUI interface
    '''

    # Check for key presses
    #key = cv2.waitKey(1)
    
    if 'adjusted_scaling' not in globals():
        adjusted_scaling = 1.0  # Default scale if not defined

    # Press 't' to toggle drawing mode
    if key == ord('t'):
        draw_box_mode = not draw_box_mode 
        print(f"Drawing mode {'enabled' if draw_box_mode else 'disabled'}")

    # Press 'c' to confirm the box
    elif key == ord('c') and final_box_coords:  
        saved_box_coords = final_box_coords
        final_box_coords = None  # Clear the current box to hide it
        print(f"Saved box coordinates: {saved_box_coords}")

    # Press 'o' to open the CAD mesh window
    # elif key == ord('o'):  
    #     display_cad_mesh(cad_mesh)
        
    # Press 'p' to project the CAD mesh in camera frame
    elif key == ord('p') and saved_box_coords is not None:  
        project_and_print_center(cad_mesh, camera_matrix, dist_coeffs)

        if 'selected_face_normal' in globals():
            # Create a rotation matrix based on the selected face normal
            up_vector = np.array([0, 0, 1]) 
            rotation_axis = np.cross(up_vector, selected_face_normal)
            rotation_angle = np.arccos(np.clip(np.dot(up_vector, selected_face_normal), -1.0, 1.0))  # Handle numerical stability

            # Create the rotation matrix using Rodrigues' rotation formula
            rotation_matrix, _ = cv2.Rodrigues(rotation_axis * rotation_angle)

            # Transform the mesh vertices based on the selected face orientation
            cad_mesh_vertices = np.asarray(cad_mesh.vertices) @ rotation_matrix.T
        else:
            cad_mesh_vertices = np.asarray(cad_mesh.vertices)

        # Scale the CAD mesh based on the tag size
        cad_bounds = np.asarray(cad_mesh.get_axis_aligned_bounding_box().get_extent())
        cad_size = np.linalg.norm(cad_bounds)
        scaling_factor = tag_size / cad_size
        cad_mesh_vertices *= scaling_factor

        # Translate the scaled CAD mesh to the AprilTag's detected position
        translation_vector = np.array([tvec.flatten()[0], tvec.flatten()[1], tvec.flatten()[2]])
        cad_mesh_vertices += translation_vector

        # Adjust z-axis based on the height of the mesh
        min_z_idx = np.argmin(cad_mesh_vertices[:, 2])  # Index of the vertex with the smallest z-coordinate = bottom of mesh
        min_z = cad_mesh_vertices[min_z_idx, 2]
        translation_z = -min_z  # Shift to align the bottom of the mesh with z = 0
        cad_mesh_vertices[:, 2] += translation_z

        # Adjust the translation based on the height of the CAD mesh
        height = np.max(cad_mesh_vertices[:, 2]) - np.min(cad_mesh_vertices[:, 2])  # Calculate the height of the mesh
        shift_height = height / 2 
        cad_mesh_vertices[:, 2] -= shift_height  # Move the CAD mesh down by half its height

        # Align the CAD mesh center to the AprilTag position
        cad_center = np.mean(cad_mesh_vertices, axis=0)
        translation_xy = np.array([tvec.flatten()[0] - cad_center[0], tvec.flatten()[1] - cad_center[1], 0])
        cad_mesh_vertices += translation_xy

        cad_center = np.mean(cad_mesh_vertices, axis=0)  # Recalculate the center of the mesh
        cad_mesh_vertices -= cad_center
        #print(f"cad mesh location: {cad_center}")

        cad_mesh.scale(adjusted_scaling, cad_center)
        print(f"Adjusted scaling: {adjusted_scaling}")

        cad_mesh.vertices = o3d.utility.Vector3dVector(cad_mesh_vertices)

    # Press 'r' to randomize mesh location and store location information
    elif key == ord('r') and saved_box_coords:

        # randomize_rotation_matrix = np.array([
        #     [0, -1, 0],
        #     [1,  0, 0],
        #     [0,  0, 1]
        # ])

        if check_size(cad_mesh, box_width, box_height):
            # Reset CAD mesh to the initial position (0, 0, 0)
            cad_mesh_vertices = np.asarray(cad_mesh.vertices)
            cad_center = np.mean(cad_mesh_vertices, axis=0)  # Get the current center of the mesh
            cad_mesh_vertices -= cad_center  # Reset mesh to 0, 0, 0 by shifting its center to origin

            #cad_mesh_vertices = cad_mesh_vertices @ randomize_rotation_matrix.T  # Rotate the mesh randomly

            random_x = np.random.uniform(0, box_width)
            random_y = np.random.uniform(0, box_height)

            # Move the CAD mesh by the width and height
            cad_mesh_vertices[:, 0] += random_x  # Move X by box width
            cad_mesh_vertices[:, 1] += random_y  # Move Y by box height

            # Update mesh with new position
            cad_mesh.vertices = o3d.utility.Vector3dVector(cad_mesh_vertices)
            print(f"Moved CAD mesh to new location: {np.mean(np.asarray(cad_mesh.vertices), axis=0)}")

            # Record info (where the randomized positions are) into a .csv file
            store_center = np.mean(np.asarray(cad_mesh.vertices), axis=0)
            save_position_to_csv(store_center, csv_filename, mesh_name)
        else:
            print("The bounding box size is too small to fit the object.")

    # Press '-' to scale mesh down
    elif key == ord('-') and key_pressed != '-':
        adjusted_scaling -= 0.05
        #adjusted_scaling = max(0.001, adjusted_scaling)
        key_pressed = '-'
        print(f"Adjusted scaling: {adjusted_scaling}")

    # Press '=' to scale mesh up
    elif key == ord('=') and key_pressed != '=':
        adjusted_scaling += 0.05
        #adjusted_scaling = min(10, adjusted_scaling)
        key_pressed = '='
        print(f"Adjusted scaling: {adjusted_scaling}")

    # Press '[' to rotate mesh on the same face
    elif key == ord('['):
        cad_mesh.center = np.mean(np.asarray(cad_mesh.vertices), axis=0)
        box_rotation_matrix = np.array([
            [-1, 0, 0],
            [0,  -1, 0],
            [0,  0, 1]
        ], dtype=np.float32)
        box_3d = box_3d @ box_rotation_matrix.T

    # Press ']' to rotate mesh on the same face
    elif key == ord(']'):
        cad_mesh.center = np.mean(np.asarray(cad_mesh.vertices), axis=0)
        box_rotation_matrix = np.array([
            [-1, 0, 0],
            [0,  -1, 0],
            [0,  0, 1]
        ], dtype=np.float32)
        box_3d = box_3d @ box_rotation_matrix.T

    # Press 'f' to rotate mesh between all its "faces"
    elif key == ord('f'):
        cad_mesh.center = np.mean(np.asarray(cad_mesh.vertices), axis=0)
        box_rotation_matrix = np.array([
            [-1, 0, 0],
            [0,  -1, 0],
            [0,  0, 1]
        ], dtype=np.float32)
        box_3d = box_3d @ box_rotation_matrix.T
        
    # If the key is released, reset key_pressed
    elif key == 255:  # No key pressed
        key_pressed = None

    # Press 'q' to exit
    elif key == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()