# Credit Risk Assesment

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from swiftmltoolz import plot_corr_heatmap, print_mutual_information
from xgboost import XGBClassifier
import pickle

# loading dataset
df = pd.read_csv('credit_risk_dataset.csv')
df_credit = df.copy()

# dropping duplicates
df_credit.drop_duplicates(inplace=True)
df.drop_duplicates(inplace=True)

# defining na columns
na_columns = ['loan_int_rate', 'person_emp_length']

# splitting columns into numerical and categorical
num_cols = df_credit.dtypes[df_credit.dtypes != 'object'].index
cat_cols = df_credit.dtypes[df_credit.dtypes == 'object'].index

#  Data Preprocessing
# dealing with na

df_cred = df.copy()
df_cred.loan_int_rate = df_cred['loan_int_rate'].fillna(
    df_cred.loan_int_rate.mean())
df_cred.person_emp_length = df_cred['person_emp_length'].fillna(
    df_cred.person_emp_length.mean())

# dealing with outliers
df_cred = df_cred[df_cred.person_age <= 120]
# feature engineering
df_cred['credit_start_year'] = df_cred.person_age - \
    df_cred.cb_person_cred_hist_length
df_cred.drop(columns=['cb_person_cred_hist_length',
             'person_age'], inplace=True)

# overwrite num_cols
num_cols_2 = ['person_income', 'person_emp_length', 'loan_amnt',
              'loan_int_rate', 'loan_percent_income', 'credit_start_year']

# Defining Ordinal and One hot encoding columns
ordinal = ['loan_grade']
one_hot = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']

# making pipelines
one_hot_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encode", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])
ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal_encode", OrdinalEncoder(
        categories=[['A', 'B', 'C', 'D', 'E', 'F', 'G']])),
    ("scalar", StandardScaler())
])

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scalar", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols_2),
    ("ord", ordinal_pipeline, ordinal),
    ("one", one_hot_pipeline, one_hot)
], n_jobs=-1, remainder='passthrough')


# splitting dependent and independent variables
X = df_cred[['person_income', 'person_home_ownership', 'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
             'credit_start_year']]
y = df_cred.loan_status
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=2)

# applying preprocessing
preprocessor.set_output(transform="pandas")
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Model Selection and Training. Goal -----> minimizing FNs, maximizing Recall
# training xgbclassifier model
ratio = float(sum(y_train == 0)) / sum(y_train == 1)
xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=ratio,  # for imbalanced data
    use_label_encoder=False
)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}
gridsearch_xgb = GridSearchCV(
    xgb,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2      # shows progress
)

print("fitting the model.....")
gridsearch_xgb.fit(X_train_preprocessed, y_train)
print("")
print("model Fitted....")
print("roc_auc_score ----->", roc_auc_score(y_test,
      (gridsearch_xgb.best_estimator_.predict_proba(X_test_preprocessed)[:, 1] > 0.31).astype(int)))

# saving the model
artifacts = {
    "preprocessor": preprocessor,
    "model": gridsearch_xgb.best_estimator_,
    "threshold": 0.31
}

with open('credit_risk_model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
