# Prostate Cancer Detection in MRI, using Clinical Latents in a U-Net (CLU-Net)

This repository contains a modified version of the baseline picai challange model. It was developed by Group 11 in the AI in Medical Imaging (2024) course. 

The group consists of 
- Ella Has
- Martin Kent Kraus
- Robert Michel
- Giorgio Nagy



## Acknowledgments

This code heavily relies on the code provided as a baseline by the [PI-CAI challenge](https://pi-cai.grand-challenge.org/). Most of the code was not changed. 
We thank our supervisor Joeran Bosma for his help brainstorming ideas, figuring out bugs, and generally supporting the process. 

## Changes

### Loss function - focal.py

### Architecture - unets.py

### Clinical variables extraction and pre-processing - image_reader.py, 

### passing clinical data to the model - data_generator.py and image_reader.py


- two additional files provided Submission.zip (RENAME) which contains the process.py for submitting code trained on this baseline to the grand challange, and regression.py, which trains a regression model using the clinical variables, but does not use them in the unet. 
## How to use
- the included process.py can be used for submitting on the grand challange. 
- This picai baseline can be installed by using 



### Installation
This picai baseline can be installed by using the following command:


```bash
pip install git+https://github.com/MKentKraus/picai_baseline
```

For more information on the general setup, folder structure, data usage or data-preprocessing, please consult the [PI-CAI baseline](https://github.com/DIAGNijmegen/picai_baseline/blob/main/README.md) readme file. 

