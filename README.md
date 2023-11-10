# Cervical Cancer Prediction

![Alt text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/Cancer%20Awareness%20Months%20heading%20image%20651x342.png)

## Business Problem

Females in Africa have a notable prevalence of cervical cancer cases, according to the "Accurate Diagnosis Health" clinic. They want to set up neighborhood screening facilities throughout Africa in order to improve their clinic's standing and address this important health issue. The identification and referral process for possible instances of cervical cancer will be improved at these centers by using an automated tool for preliminary analysis. The main goal is to guarantee quicker and more precise identification, which will allow for improved cervical cancer case evaluation and treatment.

## Data Source

I found a dataset in Kaggle named “Multi Cancer Dataset” which contains many images of different types of cancers such as Acute Lymphoblastic Leukemia, Brain Cancer, Breast Cancer, Cervical Cancer, Kidney Cancer, Lung, Colon Cancer, Lymphoma, and Oral Cancer. However, I will be focusing on Cervical Cancer since “AD Health” main goal is to determine by screening diagnosis.

This 25,000-image dataset on cervical cancer which includes these characteristics:

- Having a balance data.
- Images size : 150 x 150
- Having 5 different classes:
    - 0: Dyskeratotic
    - 1: Koilocytotic
    - 2: Metaplastic
    - 3: Parabasal
    - 4: Superficial-Intermediat

## Modeling with Convolutional Neural Networks (CNN)

### Base Model CNN Arquitecture

- Input Layer 
- 1 layer of 2D-convolutional 
- 1 layer of MaxPooling2D 
- Flattening Layer
- Dense Layers
- Output Layer

### Accuracy and Loss Curves Base Model CNN
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/Baseline%20Curves.png)

The model accuracy on the training is a 98% and the accuracy on the test data is 91%, having a validation loss of 0.3203.
The accuracy is pretty good. However, the difference in performance between the training and test data strongly indicates overfitting. To better improvement in to overfitted we can later add some dropuot and regularization layers.

As our metric for this problem we are using accuracy, but also I'm having in mind the metric recall. Want to decrease the recall metric as posible, because reducing recall, is reducing false negatives.

### Best Model CNN Arquitecture
- Input Layer (Normalization)
- 5 layers of 2D-convolutional 
- 4 layers of MaxPooling2D 
- Flattening Layer
- Dense Layers (with Regularization and Dropout)
- Output Layer

### Accuracy and Loss Curves Best Model CNN
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/Best%20Model%20So%20Far%20Curves.png)

Among the models we've tried, the third one is the best. It's really good at learning from the training data (98% accurate) and also performs well when dealing with new, unseen data (97% accurate). This model doesn't have the problem of learning the training data too well and manages to do about 6% better than the previous models. Also, it's better at recognizing different things in each group than before. Overall, this model is the strongest one we've tested.

### Looking at Recall 

In the basic model, the recall scores, which show how well the model identifies different categories, were generally good. Here are the recall scores for each category:

Dyskeratotic: 95% 
Koilocytotic: 82% 
Metaplastic: 95% 
Parabasal: 97% 
Superficial-Intermediate: 86% 

The best model, however, showed improvements in recognizing these categories. Here are the recall scores for the best model:

Dyskeratotic: 93% 
Koilocytotic: 92% 
Metaplastic: 100% 
Parabasal: 98% 
Superficial-Intermediate: 99% 

Overall, the best model did a better job at correctly recognizing or categorizing the different groups compared to the basic model. The improvements were particularly notable in identifying Metaplastic and Superficial-Intermediate categories, achieving perfect or near-perfect scores.

# Transfern Learning 
Transfer learning is a powerful technique in machine learning. By utilizing pre-trained models like VGG16, which is a well-known as convolutional neural network architecture with 16 layers, that achieve improved performance. In order to start we will download the vgg16 pre-trained and add it to the classifier and freezing all layers in order to see results with all layer freeze. In this case the structure of my classifier look like this: 

- vgg16_model = Sequential()
- vgg16_model.add(VGG16_base)
- vgg16_model.add(Lambda(lambda x: x / 255.0))
- vgg16_model.add(Flatten())
- vgg16_model.add(Dense(256, activation='relu'))
- vgg16_model.add(Dense(5, activation='softmax'))

Getting results on the train accuracy of 96% and test accuracy of 95%, where the loss is 19.36%
  
## Fine-Tuning
For a better performance i will perfom a fine-tuning in the vgg16. This process of funi-tuning involves in freezing certain layers and letting other layers be modified to gain better performance. In this case, I will freeze every other layer except layers ['block1_conv1', 'block1_conv2'] from VGG16 model and runinng it again to see results.

Achieved a higher training accuracy of 99.66% and a test accuracy of 98.92%, having reduced the test loss to 3.62%.

## Confusion Matrix Best Model Overall
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/VGG16%20CM.png)

Recall scores for Best Model Overall
 Dyskeratotic:  99%        
 Koilocytotic:  97%          
 Metaplastic:   99%           
 Parabasal:    100%           
 Superficial-Intermediat: 100%

The improvements in recall for Dyskeratotic, Metaplastic, Parabasal, and Superficial-Intermediate categories suggest that the fine-tuning of the VGG16 model has enhanced the model's performance in recognizing these classes.

# Recommendations
- Implement an automated tool, based on the fine-tuned VGG16 model, in neighborhood clinics across Africa. To enhance the identification and referral process for possible instances of cervical cancer, contributing to quicker and more precise diagnoses.
- Promote Public Awareness by conducting awareness campaigns to inform communities about screening centers, emphasizing early detection benefits.
  
# Next Steps
- Explore Various Transfer Learning Models
- Add more types of cancer, for model to learn to distinguish between different types of cancer.
