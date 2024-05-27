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
We changed the implementation of the [U-Net architecture](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/neural_networks/unets.py#L166-L194). Both the down-sampling and up-sampling paths remain unchanged, with the same dimensionalities and number of channels. We have changed the bottleneck layer at the bottom of the U-Net to output one less channel. It is replaced by a channel containing clinical information, which are upscaled through a single linear layer. The results of this linear layer are then appended to the output of the bottleneck to restore the original dimensionality. In order to be able to insert this linear layer, the definition of the unet architecture was un-recursified, and the [forward pass](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/neural_networks/unets.py#L296-L331) is done through calling the individual layers, rather than a single NN.sequential block. In addition to the segmentation map, the forward pass also outputs a confidence score. This is done in order to train with the updated loss function, which is desirable in order to be able to compare our second experiment of using the clinical variables in a logistic regression to predict the presence of cancer. 



### Clinical variables extraction and pre-processing - image_reader.py, 

### passing clinical data to the model - data_generator.py and image_reader.py


## How to use
- regression.py, which trains a regression model using the clinical variables, but does not use them in the unet. 

### Installation
This picai baseline can be installed by using the following command:


```bash
pip install git+https://github.com/MKentKraus/picai_baseline
```

For more information on the general setup, folder structure, data usage or data-preprocessing, please consult the [PI-CAI baseline](https://github.com/DIAGNijmegen/picai_baseline/blob/main/README.md) readme file. 

### Submission to the grand challenge 
We include a zipped folder, based on the provided U-Net [Grand Challenge algorithm](https://github.com/DIAGNijmegen/picai_unet_gc_algorithm). Once you have trained your model, replace the weights in the weight folder with your own. Build the docker and then hand it in on the [PI-CAI challenge](https://pi-cai.grand-challenge.org/). This zipped file contains a modified process.py which accounts for the unet outputing both the segmentation map as well as a confidence score. 

