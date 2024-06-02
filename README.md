# Prostate Cancer Detection in MRI, using a U-Net and Logistic Regression

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
The loss function is the the binary cross-entropy loss of the case-level prediction.

### Architecture - unets.py
We kept the general implementation of the [U-Net architecture](https://github.com/MKentKraus/picai_baseline/blob/logistic_regression/src/picai_baseline/unet/training_setup/neural_networks/unets.py) and added a [Logistic Regression Model](https://github.com/MKentKraus/picai_baseline/blob/93d8d35bb5a9fca98e5d9924993c5c28efabdae4/src/picai_baseline/unet/training_setup/neural_networks/unets.py#L326-L332) with a 5x1 linear layer with sigmoid activation. The Logistic Regression model takes the maximal value of in the detection map that is output by the baseline U-Net and the four clinical variables as input to the linear layer.



### Clinical variables extraction and pre-processing - image_reader.py
The functions [read\_meta\_data](https://github.com/MKentKraus/picai_baseline/blob/93d8d35bb5a9fca98e5d9924993c5c28efabdae4/src/picai_baseline/unet/training_setup/image_reader.py#L93), [log\_values](https://github.com/MKentKraus/picai_baseline/blob/93d8d35bb5a9fca98e5d9924993c5c28efabdae4/src/picai_baseline/unet/training_setup/image_reader.py#L144), and [fill\_in\_missing](https://github.com/MKentKraus/picai_baseline/blob/93d8d35bb5a9fca98e5d9924993c5c28efabdae4/src/picai_baseline/unet/training_setup/image_reader.py#L163) are used to gather the clinical data. The function read\_meta\_data retrieves the clinical data from image meta data. The function fill\_in\_missing computes missing values where possible, and otherwise fills in the median for that variable. The medians have been pre-computed in \[reference Jupyter notebook\](https://colab.research.google.com/drive/1L-ugcHQaxPheLqZSYFTXQHglrtq9gNpx?usp=sharing). Finally, log\_values takes the log of one plus the clinical variables of which taking a log results in a normal distribution.

### passing clinical data to the model - data_generator.py and image_reader.py
To pass the clinical data to the model, the following code is changed: [getting an item from the dataset](https://github.com/MKentKraus/picai_baseline/blob/93d8d35bb5a9fca98e5d9924993c5c28efabdae4/src/picai_baseline/unet/training_setup/image_reader.py#L203-L207), [add clinical data to dictionary in generator](https://github.com/MKentKraus/picai_baseline/blob/93d8d35bb5a9fca98e5d9924993c5c28efabdae4/src/picai_baseline/unet/training_setup/data_generator.py#L71), and [pass clinical data to the model and receive detection map and case-level prediction](https://github.com/MKentKraus/picai_baseline/blob/93d8d35bb5a9fca98e5d9924993c5c28efabdae4/src/picai_baseline/unet/training_setup/callbacks.py#L131-L132).

## How to use
- regression.py, which trains a regression model using the clinical variables, but does not use them in the unet. 

### Installation
This picai baseline can be installed by using the following command:


```bash
pip install git+https://github.com/MKentKraus/picai_baseline@logistic_regression
```

For more information on the general setup, folder structure, data usage or data-preprocessing, please consult the [PI-CAI baseline](https://github.com/DIAGNijmegen/picai_baseline/blob/main/README.md) readme file. 

### Submission to the grand challenge 
We include a zipped folder, based on the provided U-Net [Grand Challenge algorithm](https://github.com/DIAGNijmegen/picai_unet_gc_algorithm). Once you have trained your model, replace the weights in the weight folder with your own. Build the docker and then hand it in on the [PI-CAI challenge](https://pi-cai.grand-challenge.org/). This zipped file contains a modified process.py which accounts for the unet outputing both the segmentation map as well as a confidence score. 

