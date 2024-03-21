import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from scipy.stats import expon, reciprocal

# Carica i dati
df = pd.read_excel('/Users/guglielmo/Desktop/CODE/DataLAB/Aigab/dativ1.5.xlsx', sheet_name='TRAIN')

X = df.drop(['RACCOMANDAZIONE'], axis=1)
y = df['RACCOMANDAZIONE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisci la distribuzione dei parametri per SVC
param_distribution = {
    'C': reciprocal(0.1, 1000),  # Reciprocal è spesso usato per 'C'
    'kernel': ['linear', 'rbf'],  # Tipi di kernel
    'gamma': expon(scale=1.0)  # Expon è usato per 'gamma'
}

# Configura RandomizedSearchCV
random_search_svc = RandomizedSearchCV(SVC(random_state=42), param_distributions=param_distribution, n_iter=100, verbose=3, cv=5, random_state=42, n_jobs=-1)

# Addestramento del modello usando RandomizedSearchCV per trovare i migliori parametri
random_search_svc.fit(X_train, y_train)

# Stampa i migliori parametri trovati
print(f"Migliori parametri: {random_search_svc.best_params_}")

# Valutazione del modello con i migliori parametri sul set di test
y_pred_opt = random_search_svc.predict(X_test)
print(accuracy_score(y_test, y_pred_opt))
print(classification_report(y_test, y_pred_opt))

# Chiede all'utente se desidera salvare il modello
salva_modello = input("Vuoi salvare il modello? (sì/no): ").strip().lower()
if salva_modello == "sì" or salva_modello == "si":
    nome_modello = input("Inserisci il nome del file del modello (senza estensione): ").strip()
    # Assicurati che la cartella "models" esista
    if not os.path.exists('models'):
        os.makedirs('models')
    percorso_completo = os.path.join('models', nome_modello + '.joblib')
    joblib.dump(random_search_svc.best_estimator_, percorso_completo)
    print(f"Modello salvato come {percorso_completo}.")
else:
    print("Modello non salvato.")
