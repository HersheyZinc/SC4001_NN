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



## RNN
The base RNN model used consists of the RNN and a classifier. The classifier consists of a fully connected layer with 64 neurons and ReLU activation followed by an output layer with 5 neurons. The input is passed through a dense layer to expand its dimensions to the number of hidden dimensions in the RNN. The expanded signal is then fed into the RNN, and all the hidden states are concatenated and flattened into a single vector as input for the classifier. This limits the model to handling inputs of a fixed length only.

To study the effects of the different hyperparameters of an RNN model, we exhaustively trained models on different combinations of hyperparameters and obtained their F1 scores on the test dataset. These included the size of the hidden state (tested values: 16, 32, 64), the number of RNN layers (tested values: 1, 2, 3, 4), the architecture of the RNN cell (tested architectures: RNN, LSTM, GRU) and bi-directionality. In addition, skip connections and Add and Norm layers were tested.

Through testing, we found that model architecture and RNN cell type are key factors influencing the F1 score. THe figure below shows boxplots of F1 scores for all models, categorised by architecture (left) and RNN cell type (right). Compared to vanilla RNN cells, GRU and LSTM cells have higher F1 scores on average, and have less variation between them. Between the GRU and LSTM cells, the GRU cells achieve a maximum F1 score of 0.905 and a mean F1 score of 0.871, while LSTM cells achieve a maximum F1 score of 0.901 and a mean F1 score of 0.867. As LSTM cells are designed for long-term dependencies and GRU cells are not, this suggests that long-term dependencies are not critical for ECG heartbeat classification. Upon observation of model architecture, skip links and add & norm layers appear to reduce the variation in scores. This makes models become less sensitive to hyperparameters and reduces the time required for hyperparameter tuning. Moreover, the mean score for both skip link (0.867) and Add and Norm (0.869) layer models are higher than that of the normal models (0.854), suggesting that these model architectures improve classification performance.

![Boxplots of F1 scores against model architecture (left) and RNN cell type (right)](/figures/fig4.PNG)

## CNN-LSTM Hybrid
For our implementation of the CNN-LSTM hybrid model, we follow the general architecture of CNN used previously, replacing the final dense layer with an LSTM layer. The LSTM layer plays a crucial role in learning temporal dependencies across the sequence. Initially, the LSTM is configured with an input size of 32, which corresponds to the number of CNN filters, and a hidden size of 64, allowing it to produce a 64-dimensional representation at each timestep. However, as we experiment with different model configurations, the LSTM input size will dynamically change to match the varying number of CNN filters used in each test.  The final component of the CNN-LSTM model is the dense layer, which functions as a classification layer. This fully connected layer receives the hidden representation from the LSTM, encapsulating both local and temporal features learned throughout the network. With an output size of 5, this layer corresponds to a 5-class classification task, producing logits that represent the log-likelihood of the input sequence belonging to each class.

Documented in the table below, we conduct another ablation study to analyse both CNN and LSTM components. Here, we set the number of CNN layers, the number of LSTM layers and the bi-directionality of LSTM layers as our independent variables, while keeping all other components constant.

| Configurations | CNN Layers                     | LSTM Layers    | Bidirectional | Test F1 Score |
|----------------|--------------------------------|----------------|---------------|---------------|
| Baseline       | 1, [32]                      | 1              | False         | 0.916         |
| Model 2        | 2, [32, 64]                   | 1              | False         | 0.922         |
| Model 3        | 3, [32, 64, 128]              | 1              | False         | 0.979         |
| Model 4        | 3, [32, 64, 128]              | 1              | True          | 0.973         |
| Model 5        | 4, [32, 64, 128, 256]         | 1              | False         | 0.975         |
| Model 6        | 4, [32, 64, 128, 256]         | 1              | True          | 0.974         |
| Model 7        | 5, [32, 64, 128, 256, 512]    | 1              | False         | 0.972         |
| Model 8        | 5, [64, 128, 256, 512, 1024]  | 1              | False         | 0.982         |
| Model 9        | 5, [64, 128, 256, 512, 1024]  | 2              | False         | 0.983         |
| Model 10       | 5, [64, 128, 256, 512, 1024]  | 2 [128, 64]    | False         | 0.981         |


Our baseline model, consisting of only 1 CNN layer and 1 LSTM layer, already achieves a comparable score to our best-performing CNN model. This suggests that the combination of CNN and LSTM layers complement each other in the classification of ECG signals.

Our results found that increasing the depth of the CNN layers generally improves performance. Increasing CNN depth improves performance by allowing the model to learn more complex features. However, beyond around five layers, additional layers yield diminishing returns. This is due to redundancy in feature extraction and increased computational load, which hampers generalisation.

Bidirectional LSTMs, while capturing both past and future context, actually reduced performance in our case. The added complexity introduced unnecessary noise, making the model harder to train and prone to overfitting. Increasing the number of LSTM layers improves temporal processing. A second LSTM layer (as in Model 9) enhances the model’s ability to capture long-term dependencies, leading to better performance. Beyond two layers, however, performance plateaus or declines due to overfitting and difficulties in training deeper networks.

Model 9 performs the best, using five CNN layers for robust feature extraction and two LSTM layers for effective temporal processing, avoiding the inefficiencies and overfitting risks seen in more complex configurations. This model has an average F1 score of 0.983 on the test dataset. 

