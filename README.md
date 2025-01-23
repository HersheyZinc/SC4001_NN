# SC4001 Neural Networks and Deep Learning

## Introduction
Electrocardiogram (ECG) heartbeat classification is an important task in healthcare and medical diagnostics, aiming to automatically identify and categorise heartbeats from ECG signals into various types, such as normal and abnormal rhythms. Abnormal heartbeats, also known as arrhythmias, can indicate underlying conditions which require immediate medical attention, making early detection crucial. This project explores the application of neural networks on the ECG heartbeat classification task, investigating different aspects such as data preprocessing, model architecture, and model explainability. Overall, our best-performing model comprises a hybrid CNN-LSTM architecture and achieved an F1 score of 0.983 on the test data.

## Dataset
In this project, our focus is on the MIT-BIH Arrhythmia Database. It contains 48 half-hour excerpts of two-channel ambulatory ECG recordings (Moody & Mark, 2005). The dataset consists of 109446 samples, each labelled as one of the 5 categories: Non-ecotic (normal), supraventricular ectopic, ventricular ectopic, fusion, and unknown. Each sample is represented as a one-dimensional array corresponding to a single heartbeat. The data has been normalised to a range of 0–1 and zero-padded at the end to maintain uniform length.

![Example plot of ECG signal from each class](/figures/fig1.PNG)


