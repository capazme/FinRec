import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Load data
df = pd.read_excel('/Users/guglielmo/Desktop/CODE/DataLAB/Aigab/dativ1.5.xlsx', sheet_name='TRAIN')


# Separate features and target
X = df.drop(['RACCOMANDAZIONE'], axis=1)
y = df['RACCOMANDAZIONE']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a pipeline
pipeline = ImbPipeline([
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100))),
    ('oversampling', SMOTE(random_state=42)),
    ('classification', RandomForestClassifier(random_state=42))
])

# Parameters for GridSearchCV
param_grid = {
    'classification__n_estimators': [10, 2000],
    'classification__max_depth': [5, 10, 20, None],
    'classification__min_samples_leaf': [1, 2, 4, 7, 10, 20],
    'classification__min_samples_split': [2, 5, 10, 10]
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', verbose=3)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
def save_model(model, save_path='models/model.joblib'):
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, save_path)
    print(f"Model saved at {save_path}.")

# Interactive model saving
if input("Do you want to save the model? (yes/no): ").lower() in ['yes', 'y']:
    model_name = input("Enter model name (without extension): ").strip()
    save_model(best_model, f'models/{model_name}.joblib')
else:
    print("Model not saved.")