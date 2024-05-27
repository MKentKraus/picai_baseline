# Prostate Cancer Detection in MRI, using Clinical Latents in a U-Net (CLU-Net)

This repository contains a modified version of the baseline picai challenge U-Net. It was developed by Group 11 in the AI in Medical Imaging (2024) course. 

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
The loss function is the sum of the original loss in the baseline and the cross-entropy loss of the case-level prediction. The loss function assumes that the output of a forward pass of the model returns a tuple of (detection map, case-level prediction).

### Architecture - unets.py

### Clinical variables extraction and pre-processing - image_reader.py
The functions [read\_meta\_data](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/image_reader.py#L93), [log\_values](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/image_reader.py#L122), and [fill\_in\_missing](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/image_reader.py#L141) are used to gather the clinical data. The function read\_meta\_data retrieves the clinical data from image meta data. The function fill\_in\_missing computes missing values where possible, and otherwise fills in the median for that variable. The medians have been pre-computed in \[reference Jupyter notebook\]. Finally, log\_values takes the log of one plus the clinical variables of which taking a log results in a normal distribution.

### passing clinical data to the model - data_generator.py and image_reader.py
To pass the clinical data to the model, the following code is changed: [getting an item from the dataset](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/image_reader.py#L181C1-L183C62), [add clinical data to dictionary in generator](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/data_generator.py#L71), and [pass clinical data to the model and receive detection map and case-level prediction](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/callbacks.py#L130-L131).

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

