import cv2
import apriltag
import json
import numpy as np
import tkinter as tk
import threading
import queue
from PIL import Image, ImageTk

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AprilTag Detection with Resizable Box")
        self.canvas = tk.Canvas(root, width=800, height=600, bg='white')
        self.canvas.pack()

        self.box = None
        self.corners = []
        self.start_x = None
        self.start_y = None
        self.dragging_corner = None

        self.canvas.bind('<Button-1>', self.start_box)
        self.canvas.bind('<B1-Motion>', self.update_box)
        self.canvas.bind('<ButtonRelease-1>', self.finalize_box)

        # Load camera intrinsics
        self.load_camera_intrinsics()

        # Initialize frame queue and video thread
        self.frame_queue = queue.Queue()
        self.running = True
        self.video_thread = threading.Thread(target=self.start_video_capture)
        self.video_thread.start()

        # Start the frame update loop
        self.update_frame()

    def start_box(self, event):
        if self.box is None:
            self.start_x = event.x
            self.start_y = event.y
            self.box = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='blue')
            self.create_corners()
        else:
            # Check if the click is on a corner
            for corner in self.corners:
                if self.canvas.find_withtag(tk.CURRENT) == corner:
                    self.start_resize(event)
                    return

    def update_box(self, event):
        if self.box and not self.dragging_corner:
            self.canvas.coords(self.box, self.start_x, self.start_y, event.x, event.y)
            self.update_corners()

    def finalize_box(self, event):
        if self.box:
            x1, y1, x2, y2 = self.canvas.coords(self.box)
            print(f"Final box coordinates: ({x1}, {y1}, {x2}, {y2})")
            # Store the final coordinates as needed
        self.dragging_corner = None

    def create_corners(self):
        if self.box:
            x1, y1, x2, y2 = self.canvas.coords(self.box)
            self.corners = [
                self.canvas.create_rectangle(x1-5, y1-5, x1+5, y1+5, fill='red'),  # Top-left
                self.canvas.create_rectangle(x2-5, y1-5, x2+5, y1+5, fill='red'),  # Top-right
                self.canvas.create_rectangle(x1-5, y2-5, x1+5, y2+5, fill='red'),  # Bottom-left
                self.canvas.create_rectangle(x2-5, y2-5, x2+5, y2+5, fill='red')   # Bottom-right
            ]
            for corner in self.corners:
                self.canvas.tag_bind(corner, '<Button-1>', self.start_resize)
                self.canvas.tag_bind(corner, '<B1-Motion>', self.do_resize)
                self.canvas.tag_bind(corner, '<ButtonRelease-1>', self.end_resize)

    def update_corners(self):
        if self.box:
            x1, y1, x2, y2 = self.canvas.coords(self.box)
            self.canvas.coords(self.corners[0], x1-5, y1-5, x1+5, y1+5)
            self.canvas.coords(self.corners[1], x2-5, y1-5, x2+5, y1+5)
            self.canvas.coords(self.corners[2], x1-5, y2-5, x1+5, y2+5)
            self.canvas.coords(self.corners[3], x2-5, y2-5, x2+5, y2+5)

    def start_resize(self, event):
        self.dragging_corner = event.widget.find_closest(event.x, event.y)[0]

    def do_resize(self, event):
        if self.dragging_corner:
            x1, y1, x2, y2 = self.canvas.coords(self.box)
            if self.dragging_corner == self.corners[0]:
                self.canvas.coords(self.box, event.x, event.y, x2, y2)
            elif self.dragging_corner == self.corners[1]:
                self.canvas.coords(self.box, x1, event.y, event.x, y2)
            elif self.dragging_corner == self.corners[2]:
                self.canvas.coords(self.box, event.x, y1, x2, event.y)
            elif self.dragging_corner == self.corners[3]:
                self.canvas.coords(self.box, x1, y1, event.x, event.y)
            self.update_corners()

    def end_resize(self, event):
        self.dragging_corner = None

    def load_camera_intrinsics(self):
        # Placeholder for loading camera intrinsics
        try:
            with open('camera_intrinsics.json') as f:
                intrinsics = json.load(f)
            print("Loaded intrinsics:", intrinsics)

            if 'camera_matrix' not in intrinsics or 'distortion_coefficients' not in intrinsics:
                print("Error: Missing camera intrinsics in JSON.")
                exit(1)

            self.camera_matrix = np.array(intrinsics['camera_matrix'], dtype=np.float32)
            self.dist_coeffs = np.array(intrinsics['distortion_coefficients'], dtype=np.float32)
        except FileNotFoundError:
            print("Error: camera_intrinsics.json file not found.")
            exit(1)

    def start_video_capture(self):
        cap = cv2.VideoCapture(0)  # 0 for the default webcam
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        options = apriltag.DetectorOptions(families='tag36h11')
        self.detector = apriltag.Detector(options)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags
            detections = self.detector.detect(gray)

            # Draw detections on the frame
            for detection in detections:
                corners = detection.corners.astype(int)
                cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                # Draw the ID of the detected tag
                cv2.putText(frame, str(detection.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Convert the frame to RGB and put it in the queue for displaying
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_queue.put(frame_rgb)

        cap.release()

    def update_frame(self):
        """Update the displayed frame from the webcam."""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            # Convert the frame to a PIL Image
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)

            # Display the image on the canvas
            self.canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
            self.canvas.image = img_tk  # Keep a reference to avoid garbage collection

        self.root.after(10, self.update_frame)

    def on_closing(self):
        """Handle window closing event."""
        self.running = False
        self.root.quit()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Handle window closing
    root.mainloop()




# import cv2
# import apriltag
# import json
# import numpy as np
# import tkinter as tk
# from PIL import Image, ImageTk

# class DraggableButton:
#     def __init__(self, root):
#         self.root = root
#         self.button = tk.Button(root, text='Drag me!', width=10)
#         self.button.place(x=100, y=100)  # Initial position
#         self.button.bind('<Button-1>', self.start_drag)
#         self.button.bind('<B1-Motion>', self.do_drag)

#     def start_drag(self, event):
#         """Record the starting position when the mouse button is pressed."""
#         self.offset_x = event.x
#         self.offset_y = event.y

#     def do_drag(self, event):
#         """Move the button to follow the mouse while dragging."""
#         x = self.root.winfo_pointerx() - self.offset_x
#         y = self.root.winfo_pointery() - self.offset_y
#         self.button.place(x=x, y=y)

# if __name__ == '__main__':
#     root = tk.Tk()
#     root.title("Draggable Button Example")
#     root.geometry("800x600")  # Set the window size
#     draggable_button = DraggableButton(root)
#     root.mainloop()

# # Load camera intrinsics from a given JSON file
# try:
#     with open('camera_intrinsics.json') as f:
#         intrinsics = json.load(f)
# except FileNotFoundError:
#     print("Error: camera_intrinsics.json file not found.")
#     exit(1)

# if 'camera_matrix' not in intrinsics or 'distortion_coefficients' not in intrinsics:
#     print("Error: Missing camera intrinsics in JSON.")
#     exit(1)

# camera_matrix = np.array(intrinsics['camera_matrix'], dtype=np.float32)
# dist_coeffs = np.array(intrinsics['distortion_coefficients'], dtype=np.float32)

# # Initialize AprilTag detector
# options = apriltag.DetectorOptions(families='tag36h11')
# detector = apriltag.Detector(options)

# cap = cv2.VideoCapture(0) 

# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit(1)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect AprilTags
#     detections = detector.detect(gray)

#     for detection in detections:
#         #Draw bounding box around detected April Tag
#         corners = detection.corners.astype(int)  # Convert to integers for drawing
#         cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

#         #Draw the tag ID
#         #tag_id = detection.tag_id
#         #cv2.putText(frame, f'Tag ID: {tag_id}', (corners[0][0], corners[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     cv2.imshow('AprilTag Detection', frame)

#     # Press 'q' on keyboard to exit screen
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()