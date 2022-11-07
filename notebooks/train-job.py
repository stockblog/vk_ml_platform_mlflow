import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import infer_signature
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import catboost
from catboost import CatBoostClassifier, Pool
from catboost.utils import get_roc_curve, get_fpr_curve
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("https://ml-platform-3b36c8fb4e4070.ml.msk.vkcs.cloud/")
mlflow.set_experiment(experiment_name="driver-accident")

df = pd.read_parquet('driver-stat.parquet')

mlflow.start_run()
FEATURES = ['age', 'sex', 'car_class', 'driving_experience', 'speeding_penalties', 'parking_penalties', 'total_car_accident']
CATEGORIAL_FEATURES = ['sex', 'car_class']
TARGET = 'has_car_accident'

mlflow.log_param('features', FEATURES)
mlflow.log_param('categorial features', CATEGORIAL_FEATURES)
mlflow.log_param('target', TARGET)

train_df = df[FEATURES]
targets = df[[TARGET]]

X_train, X_test, y_train, y_test = train_test_split(train_df, targets, test_size=0.33, random_state=42)

train_pool = Pool(X_train, y_train, cat_features = CATEGORIAL_FEATURES, feature_names=FEATURES)
test_pool = Pool(X_test, cat_features = CATEGORIAL_FEATURES)

model = CatBoostClassifier(iterations=10,
                           eval_metric='F1',
                           random_seed=17,
                           train_dir = "fit_track",
                           silent=False)

mlflow.log_param('model_type', model.__class__)

grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}
mlflow.log_param('param_grid', grid)
grid_search_result = model.grid_search(grid, train_pool, verbose=False, plot=False)

mlflow.log_param('best_params', grid_search_result['params'])
y_test_pred_proba = model.predict_proba(X_test)[:, 1]
y_test_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_test_pred_proba)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

mlflow.log_metric('roc_auc', roc_auc)
mlflow.log_metric('precision', precision)
mlflow.log_metric('recall', recall)
mlflow.log_metric('f1', f1)

print(f'ROC AUC: {roc_auc}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')

input_schema = Schema([
  ColSpec("double", "age"),
  ColSpec("string", "sex"),
  ColSpec("string", "car_class"),
  ColSpec("double", "driving_experience"),
  ColSpec("double", "speeding_penalties"),
  ColSpec("double", "parking_penalties"),
  ColSpec("double", "total_car_accident")
])
output_schema = Schema([ColSpec("long", "has_car_accident")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)


mlflow.catboost.log_model(model,
                          artifact_path="driver-accident",
                          registered_model_name="driver-accident",
                          signature=signature)

mlflow.log_artifact("train-job.py")
mlflow.end_run()
