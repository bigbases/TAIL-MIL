# TAIL-MIL: Time-Aware and Instance-Learnable Multiple Instance Learning for Multivariate Time Series Anomaly Detection (AAAI 2025)

The source code for the paper "TAIL-MIL: Time-Aware and Instance-Learnable Multiple Instance Learning for Multivariate Time Series Anomaly Detection", which was presented at AAAI 2025.

## Abstract

> This study addresses the challenge of detecting anomalies in multivariate time series data. Considering a bag (e.g., multi-sensor data) consisting of two-dimensional spaces of time points and multivariate instances (e.g., individual sensors), we aim to detect anomalies at both the bag and instance level with a unified model. To circumvent the practical difficulties of labeling at the instance level in such spaces, we adopt a multiple instance learning (MIL)-based approach, which enables learning at both the bag- and instance- levels using only the bag-level labels. In this study, we introduce time-aware and instance-learnable MIL (simply, TAIL-MIL). We propose two specialized attention mechanisms designed to effectively capture the relationships between different types of instances. We innovatively integrate these attention mechanisms with conjunctive pooling applied to the two-dimensional structure at different levels (i.e., bag- and instance-level), enabling TAIL-MIL to effectively pinpoint both the timing and causative multivariate factors of anomalies. We provide theoretical evidence demonstrating TAIL-MIL's efficacy in detecting instances with two-dimensional structures. Furthermore, we empirically validate the superior performance of TAIL-MIL over the state-of-the-art MIL methods and multivariate time-series anomaly detection methods.

## Requirements
- torch
- numpy==1.21.6
- pandas==1.3.5
- tqdm==4.64.0
- sklearn==1.0.2

## Dataset
### 5-ECK-2022 (Seasonal)
- In this study, to evaluate the performance of anomaly cause identification, the [5-ECK-2022 dataset](https://data.mendeley.com/datasets/compare/m68xz4w4t9) is divided into seasonal intervals, and anomalies are labeled using the same labeling method. Additionally, further labeling is performed for the energy sources responsible for the anomalies.

### SMD Preprocessing
- The experimental dataset for this study, the Server Machine Dataset (SMD), should be imported from the link below and placed in the path 'datasets/ServerMachineDataset'.
  - link: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
  - Since we only have one example, we can only see the results of one experiment at this time.
- Once you have placed the dataset, you will need to preprocess the SMD using the commands below.
```
python preprocess.py --dataset SMD
```

## TAIL-MIL (Supervised Version)
- This is the TAIL-MIL model code that performs training with bag-level label data.

## TAIL-MIL (Surrogate Model Version)
- This is the Surrogate version of the TAIL-MIL model code, which learns based on reconstruction error.

