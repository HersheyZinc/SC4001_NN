# SC4001 Neural Networks and Deep Learning

## Introduction
Electrocardiogram (ECG) heartbeat classification is an important task in healthcare and medical diagnostics, aiming to automatically identify and categorise heartbeats from ECG signals into various types, such as normal and abnormal rhythms. Abnormal heartbeats, also known as arrhythmias, can indicate underlying conditions which require immediate medical attention, making early detection crucial. This project explores the application of neural networks on the ECG heartbeat classification task, investigating different aspects such as data preprocessing, model architecture, and model explainability. Overall, our best-performing model comprises a hybrid CNN-LSTM architecture and achieved an F1 score of 0.983 on the test data.

## Dataset
In this project, our focus is on the MIT-BIH Arrhythmia Database. It contains 48 half-hour excerpts of two-channel ambulatory ECG recordings (Moody & Mark, 2005). The dataset consists of 109446 samples, each labelled as one of the 5 categories: Non-ecotic (normal), supraventricular ectopic, ventricular ectopic, fusion, and unknown. Each sample is represented as a one-dimensional array corresponding to a single heartbeat. The data has been normalised to a range of 0–1 and zero-padded at the end to maintain uniform length.

![Example plot of ECG signal from each class](/figures/fig1.PNG)

## CNN
This section explores how modifications to the CNN architectures affect ECG heartbeat data classification performance. The figure below depicts the general architecture for our models with depth i. Each CNN block consists of a 1-dimensional convolutional layer with padding set to “same”, batch normalization, ReLU, and average pooling layer with kernel size 2. After the convolution layers, the output is flattened and passed through two fully connected layers before a softmax function as output.

![General CNN architecture for ablation study](/figures/fig2.PNG)

Our baseline model consists of just 1 convolutional layer and obtains a relatively high F1 score of 0.892. This shows that even a simple CNN architecture has the capability to learn important features of ECG heartbeats. Comparing models 1 to 5, we observe that varying the kernel size and number of filters independently does not affect F1 score significantly, while models 6 to 9 show that increasing the depth of the model yields improvement. Between models 9 and 10, we observe that modifying kernel size does not improve performance, and this is likely because the average pooling layers after each convolution reduce the temporal resolution by half. This allows convolutional layers with a smaller kernel size at higher levels to capture the same features as convolutional layers with a larger kernel size at lower levels. Lastly, we also observe that model 11 is able to maintain a comparable F1 score with models 9 and 10 despite having 32 filters for each layer. This shows that depth is a major contributing factor to classification performance, and should be prioritised over the number of filters and kernel size. Overall, we identify model 11 to be the best model as it is able to improve on the baseline model from 0.892 to 0.920 (+0.28) while having significantly fewer parameters than the other 5-layer architectures.


| Configuration | Convolution Layers [filters x kernel size] | Parameters | Test F1 Score |
|---------------|--------------------------------------------|------------|----------------|
| Model 1       | 1, [32x3]                                 | 128        | 0.892          |
| Model 2       | 1, [32x5]                                 | 192        | 0.892          |
| Model 3       | 1, [32x11]                                | 384        | 0.890          |
| Model 4       | 1, [64x3]                                 | 256        | 0.888          |
| Model 5       | 1, [128x3]                                | 384        | 0.878          |
| Model 6       | 2, [32x3, 64x3]                           | 6,336      | 0.915          |
| Model 7       | 3, [32x3, 64x3, 128x3]                    | 31,040     | 0.917          |
| Model 8       | 4, [32x3, 64x3, 128x3, 256x3]             | 129,600    | 0.917          |
| Model 9       | 5, [64x3, 128x3, 256x3, 512x3, 1024x3]    | 2,091,136  | 0.921          |
| Model 10      | 5, [64x11, 128x5, 256x3, 512x3, 1024x3]   | 2,108,032  | 0.916          |
| Model 11      | 5, [32x3, 32x3, 32x3, 32x3, 32x3]         | 12,544     | 0.920          |


### Model Explainability
We implemented Gradient-weighted Class Activation Mapping (Grad-CAM) to visualise the regions of ECG signals crucial for the model’s prediction. By computing the gradient of the model’s output with respect to the feature maps, Grad-CAM calculates the importance at each point. Visualising this in a plot allows us to infer where the model “focuses” on when it processes the ECG signal. The figure below shows samples of ECG signals with Grad-CAM values as a heatmap. The heatmap follows a sequential colour map from white to red, indicating no importance and high importance respectively. From the samples, we observed that the model places a heavy emphasis on the large spikes and drops in the ECG signal, showing that the model is making its prediction based on these highlighted parts. Unlike visualising the kernels, Grad-CAM is a more intuitive method for model explainability. This method will allow medical experts to see the areas that the CNN is focusing on, and come to their own conclusions on whether to agree or disagree with the prediction based on that information.

![Grad-CAM for CNN](/figures/fig3.PNG)
