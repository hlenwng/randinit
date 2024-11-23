# rand-init
This task initialization randomizer GUI tool takes CAD meshes, displays the object placements on the work plane, and outputs locations in a `.csv` file.
<center>
  
</center>

# Overview
Given the following inputs from the user:
- the intrinsic parameters of the camera,
- cad meshes of the objects,
- a tag on the plane of interest
  
The pipeline creates segmentation masks based on the user's labels and outputs the data as a binary .npy file, where each pixel value represents either the background (0) or the object (1). This information can be used to randomize backgrounds, enhancing a model's ability to generalize across diverse scenarios and improving its robustness to environmental variations.

# Installation
1. Clone the repository:
```
git clone https://github.com/hlenwng/rand-init.git
```
2. Install dependencies with conda:
```
conda env create -f environment.yml
conda activate rand-init
```

# Pipeline tutorial
1. Edit the example `labels.json` file with your list of object descriptions. 

   - For example, "small white ceramic mug."
2. Upload your image(s) into the `images` folder.
3. Run pipeline using:
```
python run_seg_to_binary.py
```
4. The pipeline will write 2 files:
```
/example_dir/output
    ├── output_images/img#_detections.png                     
    ├── output_binary/img#_mask.npy       
```
5. Optional: Use `check.ipynb` to visualize  `img#_mask.npy` binary file.

# Directory structure
```
/example_dir/
    ├── run_seg_to_binary.py        # Main script
    ├── labels.json                 # (Input) File containing object labels/descriptions
    ├── meshes/                     # (Input) Folder containing '.ply' meshes
    ├── output/                     # (Output) Folder to store output files
        ├── output_images/          # (Output) Folder to store output images with seg masks
        ├── output_binary/          # (Output) Folder to store output '.npy' binary files of seg masks
```
