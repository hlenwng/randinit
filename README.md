# RandInit
This task initialization randomizer GUI tool displays CAD meshes of the object(s) on a specified work plane and allows the user to randomize the location of the mesh based on specified parameters. The locations are stored in a `.csv` file.
<center>
  <img width="1320" alt="Screenshot 2024-12-15 at 10 47 25 AM" src="https://github.com/user-attachments/assets/cbc1b602-79f7-441d-a106-f6833a4a53d3" />
</center>

# Overview
Given the following inputs from the user:
- the intrinsic parameters of the camera,
- cad meshes of the objects,
- a tag on the plane of interest
  
This tool allows operators to randomize the starting positions of objects in a robot training task, increasing data collection efficiency and removing any human error and positional bias that might arise from this process. The tool stores the real-life coordinates of the object's positions (in relation to the tag), which can be used for future analysis or pipelines. This tool aims to increase a model's generalizability across various scenarios and improve its robustness to environmental variations.

# Installation
1. Clone the repository:
```
git clone https://github.com/hlenwng/rand-init.git
```
2. Install dependencies with conda:
```
conda env create -f environment.yml
conda activate randinit
```

# After the inputs are provided and the environment is installed, please view the pipeline tutorial below. 

# Pipeline tutorial
Run the tool using:
```
python rand_init_tool.py
```

1. When prompted in the terminal, enter the estimated number of demonstrations that will be run (or leave blank if unsure). An algorithm will calculate the optimal grid size given your input.

2. Press 't' to toggle drawing mode to draw a box.
3. Specify your working plane by using your mouse to drag a box starting from the April Tag. Your specified bounding box will appear as a green box.
4. Press 'c' to confirm your box.

5. Press 'p' to project your CAD mesh into the plane.
6. Press '-' or '=' to adjust the scaling of your mesh in the camera frame.

7. Press 'r' to randomize the location of your mesh in the plane. Each time you press 'r', the location will be randomized again, and the new location (3D pose with respect to the camera) will be stored in the 'mesh_positions.csv' file.
8. For more variation, press 'y' to toggle on the yaw-rotation parameter for randomization.
9. Press 'q' to quit the program.

# Directory structure
```
/example_dir/
    ├── rand_init_tool.py            # Main script
    ├── camera_intrinsics.json       # (Input) File containing camera intrinsics/parameters
    ├── meshes/                      # (Input) Folder containing '.ply' meshes
    ├── mesh_positions.csv           # (Output) Folder to store mesh locations
```
