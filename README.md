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
We changed the implementation of the [U-Net architecture](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/neural_networks/unets.py#L166-L194). Both the down-sampling and up-sampling paths remain unchanged, with the same dimensionalities and number of channels. We have changed the bottleneck layer at the bottom of the U-Net to output one less channel. It is replaced by a channel containing clinical information, which are upscaled through a single linear layer. The results of this linear layer are then appended to the output of the bottleneck to restore the original dimensionality. In order to be able to insert this linear layer, the definition of the unet architecture was un-recursified, and the [forward pass](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/neural_networks/unets.py#L296-L331) is done through calling the individual layers, rather than a single NN.sequential block. In addition to the segmentation map, the forward pass also outputs a confidence score. This is done in order to train with the updated loss function, which is desirable in order to be able to compare our second experiment of using the clinical variables in a logistic regression to predict the presence of cancer. 



### Clinical variables extraction and pre-processing - image_reader.py
The functions [read\_meta\_data](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/image_reader.py#L93), [log\_values](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/image_reader.py#L122), and [fill\_in\_missing](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/image_reader.py#L141) are used to gather the clinical data. The function read\_meta\_data retrieves the clinical data from image meta data. The function fill\_in\_missing computes missing values where possible, and otherwise fills in the median for that variable. The medians have been pre-computed in \[reference Jupyter notebook\]. Finally, log\_values takes the log of one plus the clinical variables of which taking a log results in a normal distribution.

### passing clinical data to the model - data_generator.py and image_reader.py
To pass the clinical data to the model, the following code is changed: [getting an item from the dataset](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/image_reader.py#L181C1-L183C62), [add clinical data to dictionary in generator](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/data_generator.py#L71), and [pass clinical data to the model and receive detection map and case-level prediction](https://github.com/MKentKraus/picai_baseline/blob/25d21239f8e8051a78f3f8cf2fb8b1189c3c026a/src/picai_baseline/unet/training_setup/callbacks.py#L130-L131).

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

