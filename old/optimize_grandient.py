import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint

# Carica i dati
df = pd.read_excel('/Users/guglielmo/Desktop/CODE/DataLAB/Aigab/dativ1.5.xlsx', sheet_name='TRAIN')

X = df.drop(['RACCOMANDAZIONE'], axis=1)
y = df['RACCOMANDAZIONE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parametri per RandomizedSearchCV adattati per GradientBoostingClassifier
param_dist = {
    "n_estimators": randint(10, 10000),  # Ridotto il range per evitare lunghi tempi di addestramento
    "max_depth": [3, 5, 8, 10, 15, 20, None],  # Profondità massima di ciascun albero decisionale
    "min_samples_leaf": randint(1, 100),  # Numero minimo di campioni richiesti per essere una foglia
    "min_samples_split": randint(2, 500),  # Numero minimo di campioni richiesti per dividere un nodo interno
    "learning_rate": [0.01, 0.1, 0.2, 0.5]  # Riduzione del contributo di ciascun albero
}

# Modello
model = GradientBoostingClassifier(random_state=42)

# Randomized search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=1000, cv=5, verbose=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Stampa i migliori parametri
print(f"Migliori parametri: {random_search.best_params_}")

# Usa il modello ottimizzato per le previsioni
y_pred = random_search.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

