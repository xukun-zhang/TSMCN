# TSMCN: Two-stream MeshCNN for Key Anatomical Segmentation on the Liver Surface

TSMCN is a novel deep learning framework designed for key anatomical segmentation on the liver surface using 3D triangular meshes. This project extends the MeshCNN framework into a two-stream architecture, enabling more precise segmentation of key anatomical landmarks on liver meshes.

---

## Overview

TSMCN addresses the challenges of liver surface segmentation by leveraging a dual-stream approach that processes both mesh structure and feature relationships. The framework integrates edge annotations and vertex relationships to improve segmentation accuracy. We provide the core code, datasets, pre-trained models, and downstream task support to facilitate comprehensive research and application.

---

## Features

1. **Two-Stream Architecture**:
   - Extends MeshCNN to a dual-stream architecture, enhancing segmentation performance by addressing feature entanglement.

2. **Pre-Trained Models**:
   - Models trained for anatomical segmentation are available under the `TSMCN/checkpoints/debug` directory.

3. **Dataset Management**:
   - Liver mesh `.obj` files: `TSMCN/all_obj`.
   - Edge annotations: `TSMCN/seg`.

4. **Annotation Guide**:
   - Detailed documentation and scripts for constructing liver meshes and annotating edges are provided in:
     ```plaintext
     TSMCN/dataset-annotation-guide/
     ```

5. **Downstream Tasks**:
   - Core code for 3D-2D registration based on PyTorch3D, along with input examples, is available in:
     ```plaintext
     TSMCN/PyTorch3D-3D-2D Reg/
     ```

6. **Metrics**:
   - Includes implementation for commonly used evaluation metrics:
     - **Dice Score**: Measures overlap between predicted and ground truth labels.
     - **Chamfer Distance**: Computes the distance between predicted and ground truth points.

7. **Visualization**:
   - Scripts for visualizing segmentation results and annotations are provided, enabling users to analyze and compare results effectively.
   - Examples include 3D mesh visualizations, overlaying segmentation outputs on liver meshes, and highlighting key anatomical landmarks.

8. **Training and Testing Commands**:
   - Train: `python train.py`
   - Test: `python test.py`

---

## Environment Setup

To set up the environment, refer to the [MeshCNN repository](https://github.com/ranahanocka/MeshCNN). Ensure all dependencies specified in the MeshCNN documentation are installed before running this project.

---

## Data Organization

Please organize the dataset files as follows:

1. **Split Liver Mesh Files**:
   - Randomly divide the `.obj` files in `TSMCN/all_obj` into `train`, `val`, and `real_test` subsets.
   - Move the split files to:
     ```plaintext
     TSMCN/datasets/All_data/train
     TSMCN/datasets/All_data/val
     TSMCN/datasets/All_data/real_test
     ```

2. **Move Edge Annotations**:
   - Move edge annotation files from `TSMCN/seg` to:
     ```plaintext
     TSMCN/datasets/All_data/seg
     ```

3. **Move Edge-Vertex Relationship Files**:
   - Move files mapping edges to vertex indices from `TSMCN/edges` to:
     ```plaintext
     TSMCN/datasets/All_data/edges
     ```

4. **Move Edge Soft Labels**:
   - Move edge soft label files from `TSMCN/sseg` to:
     ```plaintext
     TSMCN/datasets/All_data/sseg
     ```

The final directory structure should appear as:

```plaintext
📂 TSMCN/datasets/All_data
   ├── 📂 train          # Liver mesh files for training
   ├── 📂 val            # Liver mesh files for validation
   ├── 📂 real_test      # Liver mesh files for testing
   ├── 📂 seg            # Edge annotations
   ├── 📂 edges          # Edge-vertex relationship files
   └── 📂 sseg           # Edge soft labels

```


## Other implementations

- This project is built upon the [MeshCNN](https://github.com/ranahanocka/MeshCNN) framework, which is a general-purpose deep neural network for 3D triangular meshes. We extend our sincere thanks to the creators of MeshCNN for their valuable contribution to the field.
