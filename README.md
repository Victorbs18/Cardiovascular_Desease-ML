# Pràctica Kaggle APC UAB 2022-23
##### Víctor Benito Segura
### Cardiovascular Disease

![imagen](https://user-images.githubusercontent.com/114805561/233014826-98ac629e-d4a4-4ec0-8af0-176235aef041.png)

#### URL: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
### Features

| Feature                                      | Type of feature       |  Name         |  Type unit                                        |
|----------------------------------------------|-----------------------|---------------|---------------------------------------------------|
| Age                                          | Objective Feature     | age           | int (days)                                       |
| Height                                       | Objective Feature     | height        | int (cm)                                         |
| Weight                                       | Objective Feature     | weight        | float (kg)                                       |
| Gender                                       | Objective Feature     | gender        | categorical code                                 |
| Systolic blood pressure                      | Examination Feature   | ap_hi         | int                                              |
| Diastolic blood pressure                     | Examination Feature   | ap_lo         | int                                              |
| Cholesterol                                  | Examination Feature   | cholesterol   | 1: normal, 2: above normal, 3: well above normal |
| Glucose                                      | Examination Feature   | gluc          | 1: normal, 2: above normal, 3: well above normal |
| Smoking                                      | Subjective Feature    | smoke         | binary                                           |
| Alcohol intake                               | Subjective Feature    | alco          | binary                                           |
| Physical activity                            | Subjective Feature    | active        | binary                                           |
| Presence or absence of cardiovascular disease | Target Variable       | cardio        | binary                                           |

### Objectives
The main objective of this project is perform a model to **classificate cardiovascular disease cases**. However, we have other objectives like improve our skills in **data analysis**, **data cleaning**, doing **feature engineering**, **feature selection**, **performing models and anlayzing the results**.
### Experiments
During this project we have worked with 10 models: 
|N     |MODEL                        |
|------|-----------------------------|
|1     |LOGISTIC REGRESSION          |
|2     | LINEAR SVC                  |
|3     | KNN                         |
|4     |NAIVE BAYES                  |
|5     | RANDOM FOREST               |
|6     | DECISION TREES              |
|7     |SGDCLASSIFIER                |
|8     |LINEAR DISCRIMINANT ANALYSIS |
|9     |XGBOOST                      |
|10    |PERCEPTRON                   |


### Preprocessing
Feature Engineering: 
- BMI
- Feature Transformation: gender, cholesterol, glucose and age
        
### Model
|Model                |Train Accuracy|Test Accuracy |Recall(0)|Recall(1)|Precision(0)|Precision(1)|F1(0) | F1(1) |Average precision |Elapsed Time|
|---------------------|--------------|--------------|---------|---------|------------|------------|------|-------|------------------|------------|
|Logistic Regression  |0.722         |0.723         |0.807    |0.636    |0.693       |0.764       |0.746 |0.694  |0.667             |0.4561      |    
|Linear SVC           |0.722         |0.724         |0.825    |0.62     |0.689       |0.777       |0.751 |0.69   |0.67              |21.2        |   
|KNN                  |0.7           |0.699         |0.733    |0.665    |0.69        |0.71        |0.711 |0.686  |0.638             |2.629       |   
|Naive Bayes          |0.712         |0.715         |0.863    |0.565    |0.669       |0.802       |0.754 |0.663  |0.669             |0.177       |  
|Random Forest        |0.724         |0.726         |0.809    |0.641    |0.696       |0.767       |0.748 |0.698  |0.67              |18.02       |   
|Decision Tree        |0.725         |0.725         |0.826    |0.622    |0.69        |0.779       |0.752 |0.692  |0.672             |0.6067      |  
|SGDClassifier        |0.684         |0.702         |0.706    |0.699    |0.705       |0.7         |0.705 |0.699  |0.638             |0.4458      |   
|Linear Discriminant  |0.722         |0.723         |0.826    |10.619   |0.688       |0.777       |0.75  |0.689  |0.67              |0.282       |  
|XGBoost              |0.724         |0.727         |0.803    |0.65     |0.7         |0.764       |0.748 |0.703  |0.67              |8.877       |  
|Perceptron           |0.587         |0.637         |0.709    |0.564    |0.623       |0.656       |0.663 |0.606  |0.586             |0.2554      |  

### Conclusions

We have analyzed this cardiovascular disease dataset from kaggle and we have trained different models in order to classify if a person has a cardio disease.

Unfortunately, we have not obtained a very good accuracy, despite of we have followed all the correct steps and we have created very different models. The main reason for that is that the original **dataset doesn't have attributes that are strongly correlated with the target value 'cardio'**.

In order to solve that, we have done **Feature Engineering** and we have created new attributes like BMI but it has not been enough. The feature selection has helped us to reduce possible cases of overfitting, but it has not significantly improved the accuracy either.

In our models resume, we can see that the best classifier for all features is **Random Forest and with the features systolic blood pressure (ap_hi), well_above_cholesterol and old. With accuracies of 72.9% and 72.7& respectively**. Random Forest and XGBoost are ensemble learning methods, which means that they use multiple individual models to make predictions. These individual models are decision trees used to make predictions and combine the predictions of the individual trees to produce a final prediction.

**Future work**: if we have more time to work on this dataset we will focuse on trying to create new features from our originals that have a strong relationship with the target feature 'cardio'. This will help us in the future to generate more accurate classifier models.


