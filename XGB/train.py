import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import joblib
import numpy as np
import csv
df = pd.read_csv('/tmp2/b10902005/fintech/final/dataset_2nd/new_train_ver5.csv')
df_valid = pd.read_csv('/tmp2/b10902005/fintech/final/dataset_2nd/new_valid_ver5.csv')
output_model = "final_model"
threshold = 0.25

num_rows, num_columns = df.shape
y_train = df.iloc[:, -1].values
y_valid = df_valid.iloc[:, -1].values
X_train = df.iloc[:, 1:-1].values
X_valid = df_valid.iloc[:, 1:-1].values
xgb_model = xgb.XGBClassifier(
            n_estimators=200, 
            random_state=33, 
            tree_method='auto', 
            gpu_hist=True, 
            objective='binary:logistic',
            learning_rate=0.3,
            max_depth=36,
            reg_alpha=1,
            reg_lambda=1
        )

xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, output_model)
y_probabilities = xgb_model.predict_proba(X_valid)
y_pred = np.where(y_probabilities[:, 1] > threshold, 1, 0)

accuracy = accuracy_score(y_valid, y_pred)
conf_matrix = confusion_matrix(y_valid, y_pred)
classification_rep = classification_report(y_valid, y_pred)
print('--------------------------------------------------')
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
