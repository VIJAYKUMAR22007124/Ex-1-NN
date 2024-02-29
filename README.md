<H3>B VIJAY KUMAR</H3>
<H3>212222230173</H3>
<H3>EX. NO.1</H3>
<H3>29.02.2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.

## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
#### INCLUDE NECESSARY PACKAGES AND LIBRARIES:
```
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
#### READ THE DATASET :
```
df= pd.read_csv('Churn_Modelling (2).csv')
df
```
#### PERFORM BASIC OPERATIONS(FILLNA,ISNULL,DUPLICATED,ETC,):
```
df.isnull().sum()
df.fillna(df.mean().round(1),inplace=True)
df.duplicated()
```
#### DROP FEATURES WITH STRING VALUES :
```
df.drop('Geography',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
```
#### SPLITTING THE SET INTO X & Y:
```
X = df.iloc[: ,:-1].values
X
y = df.iloc[:,-1].values
y
```
#### SCALING THE SET TO NORMALIZE FEATURES
```
s = StandardScaler()
df1= pd.DataFrame(s.fit_transform(df))
df1
```
#### SPLITTING THE SET INTO TRAINING AND TESTING SETS:
```
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)
X_train
len(X_train)
X_test
len(X_test)
```
## OUTPUT:
#### DATASET :
![image](https://github.com/VIJAYKUMAR22007124/Ex-1-NN/assets/119657657/aeeaeca6-29d9-4edf-973a-95b9b86c5fd6)
#### NORMALIZED DATASET:
![image](https://github.com/VIJAYKUMAR22007124/Ex-1-NN/assets/119657657/3173f1a4-f151-4b52-ba41-59ce68293ae4)
#### X & Y:
![image](https://github.com/VIJAYKUMAR22007124/Ex-1-NN/assets/119657657/559f13f1-4106-4599-b958-733b299f4f1b)
#### X_TRAIN & X_TEST:
![image](https://github.com/VIJAYKUMAR22007124/Ex-1-NN/assets/119657657/a3321a7a-fbd5-4403-a4f6-4a7bf62c315b)
## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


