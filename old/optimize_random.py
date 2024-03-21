import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint

# Carica i dati
df = pd.read_excel('/Users/guglielmo/Desktop/CODE/DataLAB/Aigab/dativ1.4.xlsx', sheet_name='TRAIN')

X = df.drop(['RACCOMANDAZIONE'], axis=1)
y = df['RACCOMANDAZIONE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parametri per RandomizedSearchCV
param_dist = {
    "n_estimators": randint(10, 5000),
    "max_depth": [None, 10, 20, 30, 40, 50, 100],
    "min_samples_leaf": randint(1, 10),
    "min_samples_split": randint(2, 50),
    "bootstrap": [True, False]
}

# Modello
model = RandomForestClassifier(random_state=42)

# Randomized search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, verbose=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Stampa i migliori parametri
print(f"Migliori parametri: {random_search.best_params_}")

# Usa il modello ottimizzato per le previsioni
y_pred = random_search.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
