import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Carica i dati
df = pd.read_excel('/Users/guglielmo/Desktop/CODE/DataLAB/Aigab/dativ1.4.xlsx', sheet_name='TRAIN')

X = df.drop(['RACCOMANDAZIONE'], axis=1)
y = df['RACCOMANDAZIONE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#[CV 4/5] END bootstrap=False, max_depth=10, min_samples_leaf=4, min_samples_split=3, n_estimators=1121;, score=0.875 total time=   3.0s
#[CV 4/5] END bootstrap=True, max_depth=30, min_samples_leaf=5, min_samples_split=16, n_estimators=12064;, score=0.750 total time=  36.8s
#[CV 1/5] END bootstrap=False, max_depth=30, min_samples_leaf=5, min_samples_split=2, n_estimators=19869;, score=0.875 total time=  47.0s
#{'bootstrap': True, 'max_depth': 30, 'min_samples_leaf': 5, 'min_samples_split': 16, 'n_estimators': 3872}
# Configura il modello con i migliori parametri
from sklearn.svm import SVC

# Configura il modello SVM
model_svc = SVC(kernel='linear', C=1, random_state=42)

# Addestramento del modello
model_svc.fit(X_train, y_train)

# Valutazione del modello sul set di test
y_pred_svc = model_svc.predict(X_test)
print(accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))


# Chiede all'utente se desidera salvare il modello
salva_modello = input("Vuoi salvare il modello? (sì/no): ").strip().lower()
if salva_modello == "sì" or salva_modello == "si":
    nome_modello = input("Inserisci il nome del file del modello (senza estensione): ").strip()
    # Assicurati che la cartella "models" esista
    if not os.path.exists('models'):
        os.makedirs('models')
    percorso_completo = os.path.join('models', nome_modello + '.joblib')
    joblib.dump(model_svc, percorso_completo)
    print(f"Modello salvato come {percorso_completo}.")
else:
    print("Modello non salvato.")
