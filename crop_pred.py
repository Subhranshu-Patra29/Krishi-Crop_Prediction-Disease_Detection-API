# Importing libraries

# For ML Model training
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load dataset
PATH = 'res/Crop_recommendation.csv'
df = pd.read_csv(PATH)

# Check columns in dataset
print(df.columns)
# Check first few records
print()
print(df.head())

# Separate features from other columns of dataset
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Encode the target variable
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Get training and testing dataset
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target_encoded,test_size = 0.2,random_state =2)

# Train the XGBoost Model
XB = xgb.XGBClassifier()
XB.fit(Xtrain,Ytrain)

predicted_values = XB.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)

print("XGBoost's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values, target_names=label_encoder.classes_))


# Cross validation score (XGBoost)
score = cross_val_score(XB,features,target_encoded,cv=5)
print()
print(score)

# Dump the trained XGBoost with Pickle
XB_pkl_filename = 'res/XGBoost.pkl'
# Open the file to save as pkl file
XB_Model_pkl = open(XB_pkl_filename, 'wb')
pickle.dump(XB, XB_Model_pkl)
# Close the pickle instances
XB_Model_pkl.close()

# Save the LabelEncoder as well
with open('res/label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)


# Making predictions

data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction1 = XB.predict(data)
print(prediction1)

data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction2 = XB.predict(data)
print(prediction2)

# Inverse transform to get the original label
predicted_crop = label_encoder.inverse_transform(prediction1)

print("Predicted crop is:", predicted_crop[0])

predicted_crop = label_encoder.inverse_transform(prediction2)

print("Predicted crop is:", predicted_crop[0])