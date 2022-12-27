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

To run the CIFAR10 baseline image classification you need to run the following code:

```python
python train.py --bn --sn
``` 

### Approximation quality

You can run the following code to get results for precision of corpus decomposition for trained model from previous step.

```python
python main.py -experiment approximation_quality
``` 





### Out of distribution detection

You can run the following codes to fit SimplEx for out-of-distribution detection of CIFAR100 and SVHN.


```python
python main.py -experiment ood_detection --ood_dataset CIFAR100
python main.py -experiment ood_detection --ood_dataset SVHN

```
To get errors and plots for out-of-distribution detection. you can run the following code:

```python
python plot.py  --ood_dataset CIFAR100
python plot.py --ood_dataset SVHN

```

## Acknowledgements

This code is adapted based on the following links:

- [SimplEx](https://github.com/JonathanCrabbe/Simplex)
- [Spectral normalized Wide-resnet 28-10](https://github.com/y0ast/DUE)

## Feedback

If you have any feedback, please reach out to us at A.vahidi@campus.lmu.de

