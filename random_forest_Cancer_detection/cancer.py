import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

cancer_Df = pd.read_csv("/Users/rahmani/Documents/Assets_ML/Datasets-master/bcancer.csv")
cancer_Df = cancer_Df.drop(columns=["Unnamed: 32"])

numeric_col = cancer_Df.select_dtypes(include=[np.number])
input_col = numeric_col.columns.tolist()
target_col = "diagnosis"

cancer_train, cancer_temp, y_train, y_temp = train_test_split(cancer_Df[input_col], cancer_Df[target_col], train_size=0.7,random_state=42)
cancer_val, cancer_test, y_val, y_test = train_test_split(cancer_temp,y_temp,train_size=0.5,random_state=42)

model = RandomForestClassifier()
model.fit(cancer_train,y_train)

import matplotlib.pyplot as plt

# Get feature importances
importances = model.feature_importances_

# Match with column names
feature_names = cancer_train.columns
feat_importances = pd.Series(importances, index=feature_names)

# Sort and show top 10
print(feat_importances.sort_values(ascending=False).head(10))

# Plot
feat_importances.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.show()

y_val_pred = model.predict(cancer_val)

print("Classification report: \n", classification_report(y_val,y_val_pred))

