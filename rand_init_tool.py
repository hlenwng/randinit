import cv2
import apriltag
import json
import numpy as np
import open3d as o3d
import time
import csv
import os
from datetime import datetime
import random
import mss
import pandas as pd

'''
Values needed by user:
- JSON file with camera intrinsics
- CAD mesh file
- Tag size
'''

# Load CAD mesh using Open3D
mesh_name = 'leg_cad'
cad_mesh = o3d.io.read_triangle_mesh(f'/Users/helenwang/helen_lars/meshes/{mesh_name}.ply')
cad_mesh = cad_mesh.subdivide_midpoint(number_of_iterations=1)  # Increase iterations for a more detailed mesh
#cad_mesh = cad_mesh.simplify_quadric_decimation(target_number_of_triangles=500)  # Simplify the mesh for faster rendering
cad_mesh.compute_vertex_normals()

csv_filename = 'mesh_positions.csv'
if not os.path.isfile(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mesh_name', 'x', 'y', 'z', 'time']) 

# Initialize the session number at the start of the program
SESSION_FILE = "session_tracker.txt"
if os.path.exists(SESSION_FILE):
    with open(SESSION_FILE, 'r') as file:
        current_session_number = int(file.read().strip()) + 1
else:
    current_session_number = 1

# Save the updated session number to the file
with open(SESSION_FILE, 'w') as file:
    file.write(str(current_session_number))

# Store IRL coordinates of tag corners
tag_size = 0.06  # 6cm square
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

# Print the keyboard commands and their descriptions for the user
print('\n')
print("RandInit Tool Keyboard commands: ")
print("Press 't' to toggle drawing mode on/off")
print("Press 'c' to confirm the bounding box drawn")
print("Press 'p' to project the CAD mesh in camera frame")
print("Press '-'/'=' to scale mesh down/up")
print("Press 'r' to randomize mesh location and store location information")
print("Press 'y' to toggle yaw option for randomization")
print("Press 'q' to exit")
print("Please refer to the GIT README for more information.")
print('\n')

# Stored variables
draw_box_mode = False  # Toggle to enable drawing mode
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Initial position of box
final_box_coords = None  # Final box coordinates after button click
saved_box_coords = None  # To store the blue box coordinates after pressing 'c'
box_initialized = False # To check if the strata grid has been initialized
grid_indices = None # To store the randomized strata grid indices
n_x, n_y = 7, 7 # Size of strata grid (how "random")

cad_mesh_vertices = None  # To store projected vertices of the CAD mesh
adjusted_scaling = 1.25  # To fine-tune the scaling of displayed mesh
yaw_mode = False  # Toggle to enable yaw randomization
shift_height = 0  # Height shift for mesh alignment to z=0
translation_step = 0.1  # Translation step for mesh movement

INITIAL_KEY_DELAY = 2  # Initial delay before starting the program
KEY_DELAY = 0.5  # Delay between key presses
last_key_time = 0

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
            #print(f"Drawing box from {final_box_coords[0]} to {final_box_coords[1]}")

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

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

def csv_strata(strata_x, strata_y, filename):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare the new data
    new_data = pd.DataFrame({
        'Session': [current_session_number],
        'Strata_x': [strata_x],
        'Strata_y': [strata_y],
        'Time': [current_time]
    })

    # Append to the CSV file
    if os.path.exists(filename):
        new_data.to_csv(filename, mode='a', header=False, index=False)
    else:
        new_data.to_csv(filename, index=False)

def check_size(cad_mesh, box_width, box_height):
    cad_mesh_vertices = np.asarray(cad_mesh.vertices)
    mesh_x = np.max(cad_mesh_vertices[:, 0]) - np.min(cad_mesh_vertices[:, 0])
    mesh_y = np.max(cad_mesh_vertices[:, 1]) - np.min(cad_mesh_vertices[:, 1])

    #print(f"Mesh size: {mesh_x} x {mesh_y}")
    #print(f"Box size: {box_width} x {box_height}")

    if abs(box_width) < mesh_x or abs(box_height) < mesh_y:
        return False
    
    return True

# Create window
cv2.namedWindow('Rand Init Tool', cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback('Rand Init Tool', draw_box)
start_time = time.time()

# Open camera
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Check for key presses 
    key = cv2.waitKey(1) & 0xFF
    elapsed_time = time.time() - start_time

    # Allow keyboard commands after window has fully loaded
    # if elapsed_time < INITIAL_KEY_DELAY:
    #     continue
    
    # Add delay to prevent multiple key presses
    current_time = time.time()
    if current_time - last_key_time < KEY_DELAY:
        continue

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

            # Calculate CAD mesh scaling based on Tag size
            cad_bounds = np.asarray(cad_mesh.get_axis_aligned_bounding_box().get_extent())  # Get bounding box of CAD mesh
            cad_size = np.linalg.norm(cad_bounds)  # Calculate the size of the CAD mesh
            center = cad_mesh.get_center()

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
            cv2.imshow('Rand Init Tool', frame)

    # Display the currently drawn rectangle if coordinates are set
    if final_box_coords is not None and final_box_coords[0] != final_box_coords[1]:
        cv2.rectangle(frame, final_box_coords[0], final_box_coords[1], (0, 255, 0), 2)

    # Show the frame with AprilTags and the drawn box
    cv2.imshow('Rand Init Tool', frame)

    '''
    Check for key presses that trigger actions
    '''

    # Press 't' to toggle drawing mode
    if key == ord('t'):
        draw_box_mode = not draw_box_mode 
        print(f"Drawing mode {'enabled' if draw_box_mode else 'disabled'}")

    # Press 'c' to confirm the box
    elif key == ord('c') and final_box_coords:  
        saved_box_coords = final_box_coords
        final_box_coords = None  # Clear the current box to hide it
        print(f"Saved drawn bounding box coordinates: {saved_box_coords}")
        
        box_initialized = False
        
    # Press 'p' to project the CAD mesh in camera frame
    elif key == ord('p') and saved_box_coords is not None:  
        #project_and_print_center(cad_mesh, camera_matrix, dist_coeffs)

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

    # Press 'y' to toggle yaw randomization
    elif key == ord('y'):
        yaw_mode = not yaw_mode
        print(f"Yaw randomization {'enabled' if yaw_mode else 'disabled'}")

    # Press 'r' to randomize mesh location and store location information
    elif key == ord('r') and saved_box_coords:
        if check_size(cad_mesh, box_width, box_height):

            if not box_initialized:
                # Create and randomize strata grid
                grid_indices = [(row, col) for row in range(n_x) for col in range(n_y)]  # n_x*n_y grid
                np.random.shuffle(grid_indices)
                box_initialized = True

            # Reset CAD mesh to the initial position (0, 0, 0)
            cad_mesh_vertices = np.asarray(cad_mesh.vertices)
            cad_center = np.mean(cad_mesh_vertices, axis=0)  # Get the current center of the mesh
            cad_mesh_vertices -= cad_center  # Reset mesh to 0, 0, 0 by shifting its center to origin

            # If user enabled yaw randomization, rotate the mesh randomly around the z-axis
            if yaw_mode:
                random_yaw = np.random.uniform(np.deg2rad(1), np.deg2rad(359))
                rotation_matrix = np.array([
                    [np.cos(random_yaw), -np.sin(random_yaw), 0],
                    [np.sin(random_yaw),  np.cos(random_yaw), 0],
                    [0, 0, 1]
                ])
                cad_mesh_vertices = cad_mesh_vertices @ rotation_matrix.T

            # Grid parameters
            stratum_width = box_width / n_x
            stratum_height = box_height / n_y

            # Select a random grid index (and remove from list)
            if grid_indices:
                selected_row, selected_col = grid_indices.pop()
                #print(f"Selected stratum: ({selected_row}, {selected_col})")
            else:
                # Reset grid if all strata have been selected
                print("All strata have been used. Resetting grid.")
                box_initialized = False
                continue

            # Determine current stratum bounds
            stratum_min_x = selected_col * stratum_width
            stratum_max_x = stratum_min_x + stratum_width
            stratum_min_y = selected_row * stratum_height
            stratum_max_y = stratum_min_y + stratum_height

            # Check mesh fits in stratum
            min_bounds = cad_mesh_vertices.min(axis=0)
            max_bounds = cad_mesh_vertices.max(axis=0)
            mesh_width = max_bounds[0] - min_bounds[0]
            mesh_height = max_bounds[1] - min_bounds[1]

            # Calculate random position within the stratum
            random_x = np.random.uniform(stratum_min_x, stratum_max_x - mesh_width) - min_bounds[0]
            random_y = np.random.uniform(stratum_min_y, stratum_max_y - mesh_height) - min_bounds[1]
            #print(f"Randomized position: ({random_x}, {random_y})")

            # Translate and update the mesh
            cad_mesh_vertices[:, 0] += random_x
            cad_mesh_vertices[:, 1] += random_y
            cad_mesh.vertices = o3d.utility.Vector3dVector(cad_mesh_vertices)

            # Take a screenshot of the randomized mesh (for testing purposes)
            # timestamp = time.strftime("%Y%m%d-%H%M%S")
            # with mss.mss() as sct:
            #     screenshot = sct.shot(output=f'/Users/helenwang/downloads/randinit_{timestamp}.png')

            # Record and save position
            store_center = np.mean(np.asarray(cad_mesh.vertices), axis=0)
            save_position_to_csv(store_center, csv_filename, mesh_name)
            csv_strata(selected_row, selected_col, "strata.csv")

        else:
            print("The bounding box size is too small to fit the object.")
        
    # Press '-' to scale mesh down
    elif key == ord('-'):
        cad_mesh.scale(0.8, cad_mesh.get_center())
        adjusted_scaling *= 0.8
        print(f"Adjusted scaling: {round(adjusted_scaling, 2)}")

    # Press '=' to scale mesh up
    elif key == ord('='):
        cad_mesh.scale(1.2, cad_mesh.get_center())
        adjusted_scaling *= 1.2
        print(f"Adjusted scaling: {round(adjusted_scaling, 2)}")

    # Press 'q' to exit
    elif key == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()