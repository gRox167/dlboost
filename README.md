# mri_reconstruction_tools

## Installation
1. clone the repo
`git clone https://github.com/gRox167/mri_reconstruction_tools.git`  
3. cd into the repo directory
`cd ./mri_reconstruction_tools`
4. pip install local package
`pip install .`

if you want to install in edit mode (if you want to change any function)
`pip install -e .`

## Usage
1. import with `import mrboost`
2. there are several submodules:
```python
from mrboost import coil_sensitivity_estimation as cse
from mrboost import computation as comp
from mrboost import io_utils as iou
from mrboost import preprocessing as pre
from mrboost import reconstruction as recon
```
3. you can also refer to the notebook file (.ipynb) to find out how to use this package.
