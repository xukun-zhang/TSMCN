# Data Preparation for TSMCN

This document provides detailed instructions for organizing the dataset files in the TSMCN GitHub repository. Please follow the steps below to ensure the correct structure for training, validation, and testing.

---

## Steps for Data Organization

### 1. Randomly Split Liver Mesh Files
Navigate to the `TSMCN/all_obj` directory, which contains the liver mesh `.obj` files. Randomly divide these files into three subsets:
   - **Train**: For training the model.  
   - **Val**: For validation during training.  
   - **Real_Test**: For real-world testing.

After splitting, move the files to their respective directories under `TSMCN/datasets/All_data`:
   - Move **training files** to `TSMCN/datasets/All_data/train`.
   - Move **validation files** to `TSMCN/datasets/All_data/val`.
   - Move **testing files** to `TSMCN/datasets/All_data/real_test`.

### 2. Move Edge Annotations
The edge annotations, located in the `TSMCN/seg` directory, need to be moved to:
```plaintext
TSMCN/datasets/All_data/seg

### 3. Move Edge-Vertex Relationship Files
The files that map edges to vertex indices are stored in the `TSMCN/edges` directory. Move all these files to:
```plaintext
TSMCN/datasets/All_data/edges

### 4. Move Edge Soft Labels
The soft labels for edges are located in the `TSMCN/sseg` directory. Move these files to:
```plaintext
TSMCN/datasets/All_data/sseg
