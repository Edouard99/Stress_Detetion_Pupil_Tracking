## Table Of Contents
* [Introduction](#introduction)
* [Dataset](#dataset)
* [Data Pre-Processing](#data-pre-processing)
* [Model and Training](#model-and-training)
* [Cross Validation Results](#cross-validation-results)
* [Results on testing set](#results-on-testing-set)
* [References](#references)

## Introduction

This project aims to develop a deep learning model able to predict the emotional state (stress/no stress) of an individual based on its Pupil diameter evolution. The model has been train on a dataset built by Pedrotti et al [[1]](#1) that be found <a href="https://www.researchgate.net/profile/Marco-Pedrotti-2/publication/266613485_PUPILLARY_DATA_-_Automatic_stress_classification_with_pupil_diameter_analysis/data/543532a70cf2bf1f1f282679/data.zip?origin=publication_list">here</a>.

## Dataset

The dataset built by Pedrotti et al [[1]](#1) for their study. The aim of the study was to detect stress using wavelet transform and deep learning with pupil and electrodermal activity during driving session.

Among the measures, the dataset contains pupil size measures of 17 subjects from experimental group and 14 subjects from control group. The subjects were supposed to drive on a simulator in stressing or normal situation (control group drove in normal condition in all sessions, experimental groups drove in normal condition during the first session and then was perturbed by sound or presence of instructor during driving tasks). The pupil size was measured with a frequency of 50Hz during 80s for each subjects and each task. There was also a baseline task which was doing nothing. This conducts to 5 sessions of 80s for each subject. 

<div align="center">

| Session Number    | Conditions for experimental group           |
|:-------------:|:-------------:|
|t0 | Doing Nothing|
|t1 | Normal Driving|
|t2 | Drive With Random Sound Alarms|
|t3 | Drive With Instructors|
|t4 |Drive With Random Sound Alarms & Instructors|

</div>

The following figure is the evolution of the normalized pupil diameter as a function of time.

<p align="center">
  <img alt="ECG Sample" title="ECG Sample" src="./Media/experiment_presentation.PNG">
</p>



## Data Pre-Processing

The Preprocessing of the data is done from the <a href="https://www.researchgate.net/profile/Marco-Pedrotti-2/publication/266613485_PUPILLARY_DATA_-_Automatic_stress_classification_with_pupil_diameter_analysis/data/543532a70cf2bf1f1f282679/data.zip?origin=publication_list">raw data</a> using the notebooks <a href="./PD ds creator.ipynb">PD ds creator.ipynb</a>.

The training has been done with a cross-validation process. I extracted features from samples of 40s with a 1s step from every recording, these samples were coupled with a label : 
  * for t0 and t1 label=0 as non-stressed
  * for t2, t3 and t4 label=1 as stressed


<p align="center">
  <img alt="ECG Sample" title="ECG Sample" src="./Media/experiment_classification.PNG">
</p>


The pupil diameter (Pd) can be very different from people to people, in order to normalize my data I chose a baseline moment. I chose the first session of driving for baseline as it induces a "normal situation", not specially stressing but more than a "doing nothing situation". I computed the mean pupil diameter on the complete baseline signal. Then I computed :

$$ Pd_{normalized}(t)= Pd(t)-\overline{Pd_{Baseline}}$$

Then from each 40s (2000 points) signal, I extracted a 125 points feature vector by applying discrete wavelet transform with 4 level and with Haar's window. Finally each 125 points feature vector for learning with the average mean and average standard deviation of all the 125 vectors in the dataset. By doing this we ensure that our dataset is normalized.

<p align="center">
  <img alt="ECG Sample" title="ECG Sample" src="./Media/pre_processing.PNG">
</p>


I split the data of 29 subjects (13 from control group and 16 from experimental group) into training, validation and testing to avoid overfitting (as my features extracted from 40s samples with a sliding windows picking training and validation/testing data on a same subject would cause overfitting).
Subjects for training and validation has been permuted as I planned to use K-fold cross validation (2 subjects (1 control and 1 experimental) in validation, 27 in training (12 control and 15 experimental)), and I created 195 datasets. I selected subject 12 to be my testing subject and I never included this subject in the creation of the fold datasets.

<p align="center">
  <img alt="Kfold Datasets" title="Kfold Datasets" src="./Media/cross_validation.PNG" >
</p>

## Model and Training

My model is a Full Connected Neural Network. Each Full Connected (FC) layer is followed by a Batch Normalization layer, a Dropout(p= 0.4) layer and a LeakyRelu (a=0.2) layer. <br> The size of these layer decreases from 128 &#8594; 64 &#8594; 16 &#8594; 4 &#8594; 1. The final FC layer is followed by a Sigmoid function in order to obtain an output &#8712; [0;1]. 

The input size is 125 and the output size is 1. An output > *a-given-threshold* is considered as a stress state.

<p align="center">
  <img alt="Neural Network Architecture" title="Neural Network Architecture" src="./Media/network.PNG" >
</p>

For each fold (of the 195-fold) the model has been trained with :

&#8594; **Loss Function** = Binary Cross Entropy <br>
&#8594; **Epochs** = 250 <br>
&#8594; **Batchsize** = 256 <br>
&#8594; **Learning rate** = 0.0001 <br>
&#8594; **Optimizer** = Adam(learning rate,beta1=0.9,beta2=0.999) <br>

For each fold training the best model has been saved (based on validation set loss value) to compute the results of the cross validation.

## Cross Validation Results

Confusion Matrix used is a 2x2 confusion matrix with Stress/No stress as ground truth and Stress/No stress as prediction. This confusion matrix is computed from the validation set and the values in the confusion matrix represent a the % of the data of the validation set.

For the best model of each fold the two confusions matrixes are computed on the validation set and the average model confusion matrixes are computed.

<p align="center">
  <img alt="Average Model Confusion Matrix" title="Average Model Confusion Matrix" src="./Media/average model confusion.png">
</p>

From the best model of each fold these metrics have been computed on the validation set :

<p align="center">
  <img alt="Accuracy" title="Accuracy" src="./Media/accuracy.png">
</p><p align="center">
  <img alt="Precision" title="Precision" src="./Media/precision.png">
</p><p align="center">
  <img alt="Recall" title="Recall" src="./Media/recall.png">
</p><p align="center">
  <img alt="F1 Score" title="F1 Score" src="./Media/f1score.png">
</p>

The average metrics of my model are :

<div align="center">

| Metrics      | Mean &#177; Std|
|:-------------:|:-------------:|
|Accuracy | 0.920 &#177; 0.06|
|Precision| 0.804 &#177; 0.132|
|Recall| 0.845 &#177; 0.183|
|F1 score| 0.818 &#177; 0.148|

</div>

The WESAD paper [[1]](#1) best result on only ECG signals from chest (using Linear Discriminant Analysis) gives:

<div align="center">

| Metrics      | Mean Value|
|:-------------:|:-------------:|
|Accuracy | 0.8544|
|F1 score| 0.8131|

</div>

My Deep Learning model has an accuracy increased by **6.56%** and an f1 score increased by **0.49%**.

## Testing Results

The model has been retrained with the same process on the complete cross validation dataset (training + validation) to be tested on new data (subject 17 data).
The best model gives the following confusion matrixes for the testing set (Subject S17):

<p align="center">
  <img alt="Confusion Matrix on Testing set" title="Confusion Matrix on Testing set" src="./Media/testing confusion.png">
</p>

* **Accuracy** = 0.957
* **Precision** = 0.851
* **Recall** = 1.00
* **F1 score** = 0.920

## References
<a id="1">[1]</a> Marco Pedrotti et al. “Automatic Stress Classification With Pupil Diameter Analysis”. In:
International Journal of Human-Computer Interaction 30 (3 Mar. 2014), pp. 220–236. issn:
10447318. doi: <a href="https://doi.org/10.1080/10447318.2013.848320">10.1080/10447318.2013.848320</a> </a>.
