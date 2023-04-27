# SegmentAnything_CrackDetection

This repository contains the Python code for the SAM model and Matlab code for the U-Net model for detecting cracks in concrete structures.

# Abstract

This research assesses the performance of two deep learning models, SAM and U-Net, for detecting cracks in concrete structures. The results indicate that each model has its own strengths and limitations for detecting different types of cracks. Using the SAM's unique crack detection approach, the image is divided into various parts that identify the location of the crack, making it more effective at detecting longitudinal cracks. On the other hand, the U-Net model can identify positive label pixels to accurately detect the size and location of spalling cracks. By combining both models, more accurate and comprehensive crack detection results can be achieved. The importance of using advanced technologies for crack detection in ensuring the safety and longevity of concrete structures cannot be overstated. This research can have significant implications for civil engineering, as the SAM and U-Net model can be used for a variety of concrete structures, including bridges, buildings, and roads, improving the accuracy and efficiency of crack detection and saving time and resources in maintenance and repair. In conclusion, the SAM and U-Net model presented in this study offer promising solutions for detecting cracks in concrete structures and leveraging the strengths of both models that can lead to more accurate and comprehensive results.

# Pretrained Models
To download the pretrained models, please visit the following link: https://github.com/facebookresearch/segment-anything

# Usage
The Python code for the SAM model and Matlab code for the U-Net model are included in this repository. The code can be used to train and test the models on your own datasets. Please refer to the comments in the code for more information on how to use it.

# Requirements
The Python code requires the following packages:

PyTorch
NumPy
Matplotlib
OpenCV
The Matlab code requires the following packages:

MATLAB Deep Learning Toolbox
Citation
If you find this code useful in your research, please consider citing:

@misc{ahmadi2023application,
  title={Application of Segment Anything Model for Civil Infrastructure Defect Assessment},
  author={Mohsen Ahmadi and Ahmad Gholizadeh Lonbar and Abbas Sharifi and Ali Tarlani Beris and Mohammadsadegh Nouri and Amir Sharifzadeh Javidi},
  year={2023},
  eprint={2304.12600},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
