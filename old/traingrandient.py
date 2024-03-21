import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Carica i dati
df = pd.read_excel('/Users/guglielmo/Desktop/CODE/DataLAB/Aigab/dativ1.5.xlsx', sheet_name='TRAIN')

X = df.drop(['RACCOMANDAZIONE'], axis=1)
y = df['RACCOMANDAZIONE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#[CV 2/5] END learning_rate=0.5, max_depth=15, min_samples_leaf=1, min_samples_split=13, n_estimators=413;, score=0.875 total time=   0.5s
# Configura il modello con i migliori parametri
model_opt = GradientBoostingClassifier(
    n_estimators=413, 
    learning_rate=0.5, 
    max_depth=15,
    min_samples_leaf=1,
    min_samples_split=13, 
    random_state=42)


# Addestramento del modello
model_opt.fit(X_train, y_train)

# Valutazione del modello sul set di test
y_pred_gbc = model_opt.predict(X_test)
print(accuracy_score(y_test, y_pred_gbc))
print(classification_report(y_test, y_pred_gbc))


# Chiede all'utente se desidera salvare il modello
salva_modello = input("Vuoi salvare il modello? (sì/no): ").strip().lower()
if salva_modello == "sì" or salva_modello == "si":
    nome_modello = input("Inserisci il nome del file del modello (senza estensione): ").strip()
    # Assicurati che la cartella "models" esista
    if not os.path.exists('models'):
        os.makedirs('models')
    percorso_completo = os.path.join('models', nome_modello + '.joblib')
    joblib.dump(model_opt, percorso_completo)
    print(f"Modello salvato come {percorso_completo}.")
else:
    print("Modello non salvato.")
