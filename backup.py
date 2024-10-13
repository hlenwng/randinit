import cv2
import apriltag
import json
import numpy as np
import trimesh
import time

# Get pose estimation of AprilTag
tag_size = 0.06  # 6 cm square

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

# Load CAD mesh using trimesh
#cad_mesh = trimesh.load('/Users/helenwang/helen_lars/leg_cad.ply')
# Extract vertices of CAD mesg
#mesh_vertices = np.array(cad_mesh.vertices, dtype=np.float32)
#print("Mesh vertices:", mesh_vertices)  # Check if vertices are loaded
#limited_vertices = mesh_vertices[:10]  # Limit to first 10 vertices for speed


# Initialize AprilTag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Create interactive draggable box
drawing = False  # True if mouse is pressed
ix, iy = -1, -1  # Initial position of box
box_coords = None  # Box coordinates

box_size = np.zeros((4,3), dtype=np.float32)  # 3D points for box

def draw_box(event, x_mouse, y_mouse, flags, param):
    global ix, iy, drawing, box_coords
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_mouse, y_mouse
        box_coords = [(ix, iy), (ix, iy)]  # Start coordinates of the box

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            box_coords[1] = (x_mouse, y_mouse)  # Update the box's end coordinates

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box_coords[1] = (x_mouse, y_mouse)  # Finalize the box's end coordinates
        
        # Calculate new box dimensions based on the drawn green box
        if box_coords is not None:
            # Calculate width and height from the coordinates
            width = abs(box_coords[1][0] - box_coords[0][0]) / 100  # Convert pixels to meters (assuming 100 pixels = 1 meter)
            height = abs(box_coords[1][1] - box_coords[0][1]) / 100  # Convert pixels to meters
            
            # Update box_size with new dimensions
            box_size[:] = np.array([[width / 2, height / 2, 0],
                                     [width / 2, -height / 2, 0],
                                     [-width / 2, -height / 2, 0],
                                     [-width / 2, height / 2, 0]], dtype=np.float32)

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

    # Detect AprilTags
    detections = detector.detect(gray)

    for detection in detections:
        # Draw bounding box around detected April Tag
        corners = detection.corners.astype(int)  # Convert to integers for drawing
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

        # Get corners of tag
        img_points = detection.corners.reshape(4, 2)

        # Estimate pose using solvePnP
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

        if success:
            # Project the box size to the tag's coordinates
            box_3d_points = box_size  # Use the updated box size in meters
            box_2d_projected, _ = cv2.projectPoints(box_3d_points, rvec, tvec, camera_matrix, dist_coeffs)
            box_2d_projected = box_2d_projected.reshape(-1, 2).astype(int)

            # Draw the box that follows the AprilTag
            cv2.polylines(frame, [box_2d_projected], isClosed=True, color=(0, 0, 255), thickness=2)

            '''
            # Project CAD mesh verticies onto camera frame
            mesh_2d_projected, _ = cv2.projectPoints(mesh_vertices, rvec, tvec, camera_matrix, dist_coeffs)
            mesh_2d_projected = mesh_2d_projected.reshape(-1, 2).astype(int)

            for i in range(len(mesh_2d_projected)):
                for j in range(i+1, len(mesh_2d_projected)):
                    cv2.line(frame, tuple(mesh_2d_projected[i]), tuple(mesh_2d_projected[j]), (255, 0, 0), 1)

     '''
    # Draw the rectangle if coordinates are set
    if box_coords is not None and box_coords[0] != box_coords[1]:
        cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 255, 0), 2)
   

    # Show the frame with AprilTags and the drawn box
    cv2.imshow('April Tag Detection', frame)

    # Press 'q' on keyboard to exit screen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
