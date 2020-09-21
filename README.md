# Multi-Head CNN

This repository contains code used in the following publication:

MIDL 2020: https://openreview.net/forum?id=hRwB2BTRNu

The code in the 'src' directory contains the training-pipeline (most importantly: the model definition and the objective function) used in this work. It includes everything except for handling whole-slide-images (patch extraction, inference on WSIs, etc) which is part of a different repository that is not (yet) open source. 

An example notebook is included which demonstrates a multi-head model (training and inference) trained with the meta-loss function on the MNIST dataset. 

**abstract**

Successful clinical implementation of deep learning in medical imaging depends, in part, on the reliability of the predictions. Specifically, the system should be accurate for classes seen during training while providing calibrated estimates of uncertainty for abnormalities and unseen classes. To efficiently estimate predictive uncertainty, we propose the use of multi-head convolutional neural networks (M-heads). We compare its performance to related and more prevalent approaches, such as deep ensembles, on the task of out-of-distribution (OOD) detection. To this end, we evaluate models, trained to discriminate normal lymph node tissue from breast cancer metastases, on lymph nodes containing lymphoma. We show the ability to discriminate between the in-distribution lymph node tissue and lymphoma by evaluating the AUROC based on the uncertainty signal. Here, the best performing multi-head CNN (91.7) outperforms both Monte Carlo dropout (88.3) and deep ensembles (86.8). Furthermore, we show that the meta-loss function of M-heads improves OOD detection in terms of AUROC from 87.4 to 88.7.

