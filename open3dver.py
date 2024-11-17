import cv2
import apriltag
import json
import numpy as np
import open3d as o3d
import threading
import time
import random

tag_size = 0.06  # 6 cm square

# Load CAD mesh using Open3D
cad_mesh = o3d.io.read_triangle_mesh('/Users/helenwang/helen_lars/meshes/leg_cad.ply')
cad_mesh = cad_mesh.simplify_quadric_decimation(target_number_of_triangles=500)  # Simplify the mesh for faster rendering
cad_mesh.compute_vertex_normals()

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

# Create interactive bounding box
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Initial position of box
final_box_coords = None  # Final box coordinates after button click
saved_box_coords = None  # To store the blue box coordinates after pressing 'c'
cad_mesh_vertices = None  # To store projected vertices of the CAD mesh
selected_face_index = None  # To store the selected face index
user_choice = None
matrix_rotation = None # To store the rotation matrix based on selected face index
draw_box_mode = False  # Flag to enable drawing mode
translation_step = 0.01  # Translation step for mesh movement

def align_face_to_bottom(mesh, face_index):
    # Compute vertex normals
    mesh.compute_vertex_normals()
    
    # Get the vertices of the selected face
    triangle = np.asarray(mesh.triangles)[face_index]
    vertices = np.asarray(mesh.vertices)[triangle]
    
    # Calculate the normal vector of the selected face
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    normal /= np.linalg.norm(normal)  # Normalize the vector

    # Define the target "down" direction in the camera frame (e.g., [0, 0, -1])
    target_direction = np.array([0, 0, -1])

    # Calculate the rotation matrix to align the face normal with the target direction
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_two_vectors(normal, target_direction)
    
    # Apply rotation to the entire mesh
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    return mesh

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

def randomize_mesh_location(box_coords, z=0):
    (x_min, y_min), (x_max, y_max) = box_coords
    random_x = np.random.uniform(x_min, x_max)
    random_y = np.random.uniform(y_min, y_max)

    return np.array([random_x, random_y, z])

def update_mesh_location(mesh, new_location):
    """Translate the CAD mesh to a new location."""
    current_center = np.mean(np.asarray(mesh.vertices), axis=0)
    translation = new_location - current_center
    mesh.translate(translation)

def handle_keypress(event, x, y, flags, param):
    global cad_mesh
    if event == cv2.EVENT_KEYDOWN:
        if chr(flags & 0xFF) == 'r':  # Press 'r' to shift the mesh
            # Get current mesh center
            center = np.mean(np.asarray(cad_mesh.vertices), axis=0)
            # Shift by translation_step in the X and Y direction
            new_center = center + np.array([translation_step, translation_step, 0])
            update_mesh_location(cad_mesh, new_center)
            print(f"Shifted mesh to new location: {new_center}")

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

def display_cad_mesh(mesh):
    global user_choice, selected_face_index, matrix_rotation
    
    # Assign faces to each of the six primary directions
    directions = assign_faces_to_directions(mesh)
    reset_mesh_color(mesh)

    # Create the Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="CAD Mesh", width=800, height=600)
    vis.add_geometry(mesh)

    # Start a thread to get user input
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
            selected_color = [1.0, 0.0, 0.0]  # Red color

            # Define rotation matrices
            rotation_matrices = {
                "front": np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]),
                "back": np.array([
                    [1, 0, 0],
                    [0, np.cos(np.pi), -np.sin(np.pi)],
                    [0, np.sin(np.pi), np.cos(np.pi)]
                ]),
                "left": np.array([
                    [np.cos(np.pi / 2), 0, np.sin(np.pi / 2)],
                    [0, 1, 0],
                    [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]
                ]),
                "right": np.array([
                    [np.cos(-np.pi / 2), 0, np.sin(-np.pi / 2)],
                    [0, 1, 0],
                    [-np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]
                ]),
                "top": np.array([
                    [1, 0, 0],
                    [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2)],
                    [0, np.sin(-np.pi / 2), np.cos(-np.pi / 2)]
                ]),
                "bottom": np.array([
                    [1, 0, 0],
                    [0, np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                    [0, np.sin(np.pi / 2), np.cos(np.pi / 2)]
                ])
            }

            # Map user choice to direction and color the face
            if choice == 1:
                color_faces(mesh, directions["front"], selected_color)
                selected_face_index = directions["front"][0] # Store the selected face index
                matrix_rotation = rotation_matrices["front"]
                print("Colored the front faces.")
            elif choice == 2:
                color_faces(mesh, directions["back"], selected_color)
                selected_face_index = directions["back"][0] # Store the selected face index
                matrix_rotation = rotation_matrices["back"]
                print("Colored the back faces.")
            elif choice == 3:
                color_faces(mesh, directions["left"], selected_color)
                selected_face_index = directions["left"][0] # Store the selected face index
                matrix_rotation = rotation_matrices["left"]
                print("Colored the left faces.")
            elif choice == 4:
                color_faces(mesh, directions["right"], selected_color)
                selected_face_index = directions["right"][0] # Store the selected face index
                matrix_rotation = rotation_matrices["right"]
                print("Colored the right faces.")
            elif choice == 5:
                color_faces(mesh, directions["top"], selected_color)
                selected_face_index = directions["top"][0] # Store the selected face index
                matrix_rotation = rotation_matrices["top"]
                print("Colored the top faces.")
            elif choice == 6:
                color_faces(mesh, directions["bottom"], selected_color)
                selected_face_index = directions["bottom"][0] # Store the selected face index
                matrix_rotation = rotation_matrices["bottom"]
                print("Colored the bottom faces.")
            else:
                print("Invalid choice. Please select a number between 0 and 6.")

            # Update mesh visualization with new colors
            vis.update_geometry(mesh)
        
        # Poll events and update the renderer
        vis.poll_events()
        vis.update_renderer()
        
        # Add a slight delay to avoid high CPU usage
        time.sleep(0.01)

    vis.destroy_window()
    input_thread.join()  # Ensure input thread is cleaned up

def draw_box(event, x_mouse, y_mouse, flags, param):
    global ix, iy, drawing, final_box_coords
    
    if not draw_box_mode: # Don't allow drawing if 'f' was not pressed
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_mouse, y_mouse  # Set the initial position

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the box's end coordinates based on mouse movement
            final_box_coords = [(ix, iy), (x_mouse, y_mouse)]
            print(f"Drawing box from {final_box_coords[0]} to {final_box_coords[1]}")

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def project_and_print_center(mesh, camera_matrix, dist_coeffs):
    """Project the CAD mesh center and print 3D and 2D positions."""
    # Get the vertices of the CAD mesh
    vertices = np.asarray(mesh.vertices)
    
    # Compute the center of the mesh (mean of all vertices)
    center_3d = vertices.mean(axis=0)
    print(f"Center of CAD Mesh (3D): {center_3d}")

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
    print(f"Center of CAD Mesh (2D): {projected_center[0]}")
    return projected_center[0]


# Create window
cv2.namedWindow('April Tag Detection', cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback('April Tag Detection', draw_box)

# Open camera
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

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
            #cv2.circle(frame, tuple(projected_center), 5, (0, 255, 0), -1)  # Draw the 2D center

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
                cad_mesh = align_face_to_bottom(cad_mesh, face_index)
                print(f"Aligned face {face_index} to bottom.")

            # Calculate CAD mesh scaling based on Tag size
            cad_bounds = np.asarray(cad_mesh.get_axis_aligned_bounding_box().get_extent())  # Get bounding box of CAD mesh
            cad_size = np.linalg.norm(cad_bounds)  # Calculate the size of the CAD mesh
            scaling_factor = tag_size / cad_size  # Compute scaling factor based on tag size

            # Compute the center of the mesh
            center = cad_mesh.get_center()

            # Scale the mesh with the computed center
            cad_mesh.scale(scaling_factor * 10, center)

            # Prepare to project the CAD mesh
            if cad_mesh_vertices is not None:
                # Rotate, scale, and translate CAD mesh vertices
                scaled_vertices = np.asarray(cad_mesh_vertices) * scaling_factor * 10  # Apply scaling
                rotated_vertices = np.dot(matrix_rotation, scaled_vertices.T).T  # Apply rotation
                transformed_vertices = rotated_vertices + tvec.flatten()  # Apply translation

                # Project the transformed CAD mesh vertices onto the 2D frame
                projected_vertices, _ = cv2.projectPoints(transformed_vertices, rvec, tvec, camera_matrix, dist_coeffs)
                projected_vertices = projected_vertices.reshape(-1, 2).astype(int)

                # Draw the edges of the CAD mesh on the frame
                for i, vertex in enumerate(projected_vertices):
                    next_vertex = projected_vertices[(i + 1) % len(projected_vertices)]  # Wrap around to create a closed loop
                    # Ensure each vertex only contains two elements and convert them to integers
                    cv2.line(frame, tuple(map(int, vertex[:2])), tuple(map(int, next_vertex[:2])), (0, 0, 255), thickness=2)  # Draw red edges

            # Display the final frame
            cv2.imshow('April Tag Detection', frame)

    # Draw the currently drawn rectangle if coordinates are set
    if final_box_coords is not None and final_box_coords[0] != final_box_coords[1]:
        cv2.rectangle(frame, final_box_coords[0], final_box_coords[1], (0, 255, 0), 2)

    # Show the frame with AprilTags and the drawn box
    cv2.imshow('April Tag Detection', frame)

    '''
    Check for key presses that trigger actions
    Eventually, convert these to buttons within the GUI interface
    '''

    # Check for key presses
    key = cv2.waitKey(1)
    
    # Press 'o' to open the CAD mesh window
    if key == ord('o'):  
        display_cad_mesh(cad_mesh)

    # Press 'c' to confirm the box
    elif key == ord('c') and final_box_coords:  
        saved_box_coords = final_box_coords
        final_box_coords = None  # Clear the current box to hide it
        print(f"Saved box coordinates: {saved_box_coords}")
        
    # Press 'p' to project the CAD mesh in camera frame
    elif key == ord('p') and saved_box_coords is not None:  
        project_and_print_center(cad_mesh, camera_matrix, dist_coeffs)

        if 'selected_face_normal' in globals():
            # Create a rotation matrix based on the selected face normal
            up_vector = np.array([0, 0, 1])  # Assuming Z is up in your coordinate system
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

        # Scale the vertices of the CAD mesh
        cad_mesh_vertices *= scaling_factor

        # Translate the scaled CAD mesh to the AprilTag's detected position
        translation_vector = np.array([tvec.flatten()[0], tvec.flatten()[1], tvec.flatten()[2]])

        cad_mesh_vertices += translation_vector

    # Press 't' to toggle drawing mode
    elif key == ord('t'):
        draw_box_mode = not draw_box_mode 
        print(f"Drawing mode {'enabled' if draw_box_mode else 'disabled'}")
   
    # Press 'r' to randomize mesh location
    elif key == ord('r') and saved_box_coords:
        # random_location = randomize_mesh_location(saved_box_coords)
        # update_mesh_location(cad_mesh, random_location)
        # print(f"Randomized mesh location to {random_location}")
        center = np.mean(np.asarray(cad_mesh.vertices), axis=0)
        new_center = center + np.array([translation_step, translation_step, 0])
        update_mesh_location(cad_mesh, new_center)
        print(f"Shifted mesh to new location: {new_center}")

    # Press 'q' to exit
    elif key == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()