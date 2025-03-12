import os
import kagglehub
import pandas as pd
# for model training
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# for preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

# import model for regression
from xgboost import XGBRegressor

import joblib

# Download latest version
path = kagglehub.dataset_download("ethicalstar/loan-prediction")
# Read dataset.
df = pd.read_csv(os.path.join(path, os.listdir(path)[0]))

# Split the data into training and testing sets
X = df.drop(['Id', 'Risk_Flag'], axis=1)
y = df['Risk_Flag']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)

# Define the ColumnTransformer with three different preprocessing steps
tr1 = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [3, 4, 5, 6, 7, 8]),  # One-hot encode categorical columns
    ('std', StandardScaler(), [0]),  # Standardize numerical column(s)
    ('qtl_trsf', QuantileTransformer(output_distribution="normal"), [1, 2, 9, 10])  # Apply quantile transformation to numerical column(s)
])

# Apply the transformations to the training data
X_train_Scaled = tr1.fit_transform(X_train)

# Apply the same transformations to the testing data (using only transform, not fit_transform)
X_test_Scaled = tr1.transform(X_test)

# Train XGBoost model
params = {
        "objective": "binary:logistic",  # Binary classification
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "eta": 0.8,
    }

dtrain = xgb.DMatrix(X_train_Scaled, label=y_train)
dtest = xgb.DMatrix(X_test_Scaled, label=y_test)
model = xgb.train(params, dtrain, num_boost_round=100)

y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.3).astype(int)  # Convert probabilities to class labels
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Calculate the accuracy and F1 score.
# It should be: Accuracy: 0.89, F1 Score: 0.60
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# Save elements for inference.
# Data samples.
df[:10000].to_csv("loan-prediction-subsample.csv")

# The ColumnTransformer.
joblib.dump(tr1, 'column_transformer.pkl')
# The model.
model.save_model("loans.model")