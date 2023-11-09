# Cervical Cancer Prediction

![Alt text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/Cancer%20Awareness%20Months%20heading%20image%20651x342.png)

## Bussines Problem

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

# Modeling

## Base Model CNN Arquitecture

- Input Layer 
- 1 layer of 2D-convolutional 
- 1 layer of MaxPooling2D 
- Flattening Layer
- Dense Layers
- Output Layer

# Accuracy and Loss Curves Base Model
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/Baseline%20Curves.png)

The model accuracy on the training is a 98% and the accuracy on the test data is 91%, having a validation loss of 0.3203.
The accuracy is pretty good. However, the difference in performance between the training and test data strongly indicates overfitting. To better improvement in to overfitted we can later add some dropuot and regularization layers.

As our metric for this problem we are using accuracy, but also I'm having in mind the metric recall. Want to decrease the recall metric as posible, because reducing recall, is reducing false negatives.

## Best Model Arquitecture
- Input Layer (Normalization)
- 5 layers of 2D-convolutional 
- 4 layers of MaxPooling2D 
- Flattening Layer
- Dense Layers (with Regularization and Dropout)
- Output Layer

# Accuracy and Loss Curves Best Model
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/Best%20Model%20So%20Far%20Curves.png)

# Best Model Confusion Matrix
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/Best%20Model%20CM.png)

Among the models we've tried, the third one is the best. It's really good at learning from the training data (98% accurate) and also performs well when dealing with new, unseen data (97% accurate). This model doesn't have the problem of learning the training data too well and manages to do about 6% better than the previous models. Also, it's better at recognizing different things in each group than before. Overall, this model is the strongest one we've tested.

# Looking at Recall 

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

# Recommendations

# Next Steps















