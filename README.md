# Cervical Cancer Prediction
#### by [Gloria Pintado](https://github.com/gloriapintado)

![Alt text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/Cancer%20Awareness%20Months%20heading%20image%20651x342.png)

## Business Problem

Females in Africa have a notable prevalence of cervical cancer cases, according to the "Accurate Diagnosis Health" clinic. They want to set up neighborhood screening facilities throughout Africa in order to improve their clinic's standing and address this important health issue. The identification and referral process for possible instances of cervical cancer will be improved at these centers by using an automated tool for preliminary analysis. The main goal is to guarantee quicker and more precise identification, which will allow for improved cervical cancer case evaluation and treatment.

## Data Source
The dataset in Kaggle is named “Multi Cancer Dataset,” which contains many images of different types of cancer.
The dataset had 25,000 images.

- Having 5 classes of cells (5,000 images).
         0: Dyskeratotic \
         1: Koilocytotic \
         2: Metaplastic \
         3: Parabasal \
         4: Superficial-Intermediate
         
- Balance data 
- Image size: 150 x 150  

### Classes of Cells 

Contains normal, abnormal and benign cells.
- Superficial-Intermediate cells (normal)
- Parabasal cells (normal)
- Koilocytotic cells (abnormal)
- Dyskeratotic cells (abnormal)
- Metaplastic cells (benign)
  
Abnormal cells may indicate the presence of precancerous or cancerous cells.

# Modeling

## Base Model 

- Input Layer 
- 1 layer of 2D-convolutional 
- 1 layer of MaxPooling2D 
- Flattening Layer
- Dense Layers
- Output Layer

### Accuracy and Loss Curves Base Model 
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/Baseline%20Curves.png)

### Confusion Matrix Base Model
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/Base%20Confusion%20Matrix.png)

The model accuracy on the training is 98%, and the accuracy on the test data is 91%, with a validation loss of 32.03%.
The accuracy is pretty good. However, the difference in performance between the training and test data strongly indicates overfitting. To improve overfitting, we can later add some dropout and regularization layers.

As our metric for this problem, we are using accuracy, but I'm also keeping in mind the metric of recall. I want to decrease the recall metric as much as possible because reducing recall means reducing false negatives.

## Transfer Learning : VGG16 Model
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/VGG16_architecture.png)

This is the VGG16 neural network, pretrained on various images for image classification, offering the advantage of improved accuracy in image recognition.

### VGG16 Model before Fine-Tune

- Initially, were all layers are frozen, resulting in a 96% training accuracy and 95% test accuracy with a test loss of 19.42%. Next step is fine tuning for better results.

### VGG16 Model after Fine-Tune (Best Model)

#### Accuracy and Loss Curves Best Model
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/VGG16%20tune%20curves.png)

#### Confusion Matrix Best Model
![Alt Text](https://github.com/gloriapintado/Cervical-Cancer-Prediction/blob/main/images/VGG16%20tune%20confusion%20matrix.png)

For a better performance, I will perform a fine-tuning in the VGG16. This process of fine-tuning involves freezing certain layers and letting other layers be modified to gain better performance. In this case, I will freeze every other layer except layers ['block1_conv1', 'block1_conv2'] from the VGG16 model and run it again to see results.

Achieved a higher training accuracy of 99.69% and a test accuracy of 99.00%, having reduced the test loss to 2.07%.

#### Looking at Recall in Best Model

After the accuracy metric being important, the recall metric is also really important since considering the importance of minimizing false negatives. In the context of medical applications, such as the detection of diseases like cervical cancer, false negatives can have serious issue.

Recall scores for Best Model Overall

- Dyskeratotic: 99% 
- Koilocytotic: 97% 
- Metaplastic: 99% 
- Parabasal: 100% 
- Superficial-Intermediat: 100%

The improvements in recall for Dyskeratotic, Metaplastic, Parabasal, and Superficial-Intermediate categories suggest that the fine-tuning of the VGG16 model has enhanced the model's performance in recognizing these classes.

# Recommendations
- Implement an automated tool, based on the fine-tuned VGG16 model, in neighborhood clinics across Africa. To enhance the identification and referral process for possible instances of cervical cancer, contributing to quicker and more precise diagnoses.
- Promote Public Awareness by conducting awareness campaigns to inform communities about screening centers, emphasizing early detection benefits.
  
# Next Steps
- Explore Various Transfer Learning Models
- Add more types of cancer, for model to learn to distinguish between different types of cancer.
