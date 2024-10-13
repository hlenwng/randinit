import cv2
import apriltag
import json
import numpy as np

#Get pose estimation of AprilTag
tag_size = 0.06  #6 cm square

#Store IRL coordinates of tag corners
obj_points = np.array([
    [-tag_size / 2, -tag_size / 2, 0],
    [tag_size / 2, -tag_size / 2, 0],
    [tag_size / 2, tag_size / 2, 0],
    [-tag_size / 2, tag_size / 2, 0]
], dtype=np.float32)

#Load camera intrinsics from a given JSON file
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

#Initialize AprilTag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

#Create interactive bounding box
drawing = False  #True if mouse is pressed
ix, iy = -1, -1  #Initial position of box
box_coords = None  #Box coordinates
final_box_coords = None  #Final box coordinates after button click
scale_factor = 0.01  #Default scaling factor

def draw_box(event, x_mouse, y_mouse, flags, param):
    global ix, iy, drawing, box_coords
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_mouse, y_mouse
        box_coords = [(ix, iy), (ix, iy)]  #Start coordinates of the box

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            box_coords[1] = (x_mouse, y_mouse)  #Update the box's end coordinates

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box_coords[1] = (x_mouse, y_mouse)  #Finalize the box's end coordinates

#Create window
cv2.namedWindow('April Tag Detection', cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback('April Tag Detection', draw_box)

#Open camera
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect AprilTags
    detections = detector.detect(gray)

    for detection in detections:
        #Draw bounding box around detected AprilTag
        corners = detection.corners.astype(int)  #Convert to integers for drawing
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=4)

        #Get corners of tag
        img_points = detection.corners.reshape(4, 2)

        #Estimate pose using solvePnP
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

        if success:
            #Draw the final box based on the drawn bounding box
            if final_box_coords is not None:
                box_x1, box_y1 = final_box_coords[0]
                box_x2, box_y2 = final_box_coords[1]

                #Calculate the width and height of the drawn box in meters
                width_meters = abs(box_x2 - box_x1) * scale_factor
                height_meters = abs(box_y2 - box_y1) * scale_factor

                #Calculate the center of the drawn bounding box
                box_center_x = (box_x1 + box_x2) / 2
                box_center_y = (box_y1 + box_y2) / 2

                #Define 3D points for the final box in the same plane as the AprilTag
                red_box_3d_points = np.array([
                    [-width_meters / 2, -height_meters / 2, 0],  #Bottom-left
                    [width_meters / 2, -height_meters / 2, 0],   #Bottom-right
                    [width_meters / 2, height_meters / 2, 0],    #Top-right
                    [-width_meters / 2, height_meters / 2, 0]     #Top-left
                ], dtype=np.float32)

                #Project the 3D points of the final box to 2D
                projected_box, _ = cv2.projectPoints(red_box_3d_points, rvec, tvec, camera_matrix, dist_coeffs)
                projected_box = projected_box.reshape(-1, 2).astype(int)

                #Calculate the center of the AprilTag
                tag_center_x = np.mean(corners[:, 0])
                tag_center_y = np.mean(corners[:, 1])

                #Translate the projected final box to the center of the AprilTag
                translation_vector = np.array([tag_center_x - box_center_x, -(tag_center_y - box_center_y)])
                translated_box = projected_box + translation_vector.astype(int)

                #Draw the translated final box on the frame
                cv2.polylines(frame, [translated_box], isClosed=True, color=(255, 0, 0), thickness=2)

    #Draw the rectangle if coordinates are set ('c' is clicked)
    if box_coords is not None and box_coords[0] != box_coords[1]:
        cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 255, 0), 2)

    #Show the frame with AprilTags and the drawn box
    cv2.imshow('April Tag Detection', frame)

    #Check for key presses
    key = cv2.waitKey(1)
    
    if key == ord('c') and box_coords is not None:  #Press 'c' to confirm the box
        final_box_coords = box_coords
        box_coords = None  #Reset box_coords for future drawing
    elif key == ord('q'):  #Press 'q' to exit
        break
    elif key == ord('.'):  #Press '.' to increase the scaling factor  
        scale_factor += 0.001 
        print(f"Scaling factor increased: {scale_factor}")
    elif key == ord(','):  #Press ',' to decrease the scaling factor  
        scale_factor = max(0.001, scale_factor - 0.001)
        print(f"Scaling factor decreased: {scale_factor}")

cap.release()
cv2.destroyAllWindows()