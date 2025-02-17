# Robustness-Estimation-Experiments

This repository contains the code and neural networks from the paper: 

**Empirical Analysis of Upper Bounds for Robustness Distributions using Adversarial Attacks** 

*Author(s): Aaron Berger, Nils Eberhardt, Annelot W. Bosman, Henning Duwe, Holger H. Hoos and Jan N. van Rijn* \
Published in: [Conference/Journal Name, Year] \
citation key: 

## Overview
This repository contains:
- Pre-trained models in **ONNX** format.
- Experiment scripts and instructions for reproducing the results on **MNIST**.

## Networks
- MNIST
  - [mnist-net_256x2](networks/mnist_networks/mnist-net_256x2.onnx)
  - [mnist_relu_3_50](networks/mnist_networks/mnist_relu_3_50.onnx)
  - [mnist_relu_4_1024](networks/mnist_networks/mnist_relu_4_1024.onnx)
  - [mnist_relu_9_100](networks/mnist_networks/mnist_relu_9_100.onnx)
 
## Experiment Scripts
In the experiments folder, one can find scripts to compute robustness distributions using the following algorithms:
- [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [AutoAttack](https://github.com/fra31/auto-attack)
- [fab-attack](https://github.com/fra31/fab-attack)
- Fast Gradient Sign Method (FGSM) [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)
- Projected Gradient Descent (PGD) [Madry et al., 2018](https://arxiv.org/abs/1706.06083)


## External Packages
This project uses the following external packages:
- [VERONA](https://github.com/ADA-research/VERONA): An open-source package for creating Robustness Distributions.
