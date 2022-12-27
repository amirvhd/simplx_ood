# SN-SimplEx

This repository is for the Explaining latent representation of distance-aware model. This repository is based on the
[SimplEx](https://github.com/JonathanCrabbe/Simplex) repository. The original SImplEx is taken from the original code.

## Table of contents

* [Badges](#general-information)
* [Tasks](#Features)
* [Installation](#Installation)
* [Usage/Examples](#Usage/Examples)
* [Acknowledgements](#Acknowledgements)
* [Feedback](#Feedback)

## Badges


[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

### Dependency

![python](https://img.shields.io/badge/Python-3.8-brightgreen)
![pytorch_lightning](https://img.shields.io/badge/Pytorch_lightning-1.6.5-brightgreen)
![torch](https://img.shields.io/badge/Torch-1.12.1-brightgreen)

## Tasks

- Image classification
- Approximation quality
- Out of distribution detection


## Installation

Install requirments:

```python
pip install - r requirements.txt
```


## Usage/Examples

### Image classification

Use the Berttopic modeling for a custom dataset. Since the Berttopic has its own tokenizer, the tokenizer function in
preprocess class is commented out.

```python
python train.py
``` 

### Approximation quality

You can find the preprocessing class in Topic modeling/Utils. To use this class:

```python
python main.py -experiment approximation_quality
``` 

The preprocessing class contains 7 features:




### Out of distribution detection

This project uses Bert model as encoder and add a linear to the model for Multi-label classification for the


- ToxicCommentDataset class :  a custom dataset class that inherit torch Dataset for Toxic-comment dataset.



```python
python main.py -experiment ood_detection --ood_dataset CIFAR100
python main.py -experiment ood_detection --ood_dataset SVHN

```


## Acknowledgements

This code is adapted based on the following links:

- [SimplEx](https://github.com/JonathanCrabbe/Simplex)
- [Spectral normalized Wide-resnet 28-10](https://github.com/y0ast/DUE)

## Feedback

If you have any feedback, please reach out to us at A.vahidi@campus.lmu.de

