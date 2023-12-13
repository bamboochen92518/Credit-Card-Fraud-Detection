import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import joblib
import numpy as np
import csv

df_test = pd.read_csv('/tmp2/b10902005/fintech/final/dataset_2nd/final_test.csv')
model_name = "final_model"
threshold = 0.25

X_test = df_test.iloc[:, 1:].values
txkey = df_test.iloc[:, 0].values.tolist()
length = len(txkey)
loaded_model = joblib.load(model_name)
y_probability = loaded_model.predict_proba(X_test)
y_pred = np.where(y_probability[:, 1] > threshold, 1, 0)
length = len(y_probability)
data = list()
for i in range(length):
    data.append([txkey[i], y_pred[i]])

csv_file_path = f'prediction.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    header = ["txkey", "pred"]
    csv_writer.writerow(header)
    for row in data:
        csv_writer.writerow(row)
