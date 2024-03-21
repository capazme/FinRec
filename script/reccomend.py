import joblib
import pandas as pd
from dati import normalizza_dati_prodotti, normalizza_dati_utenti
# Percorso al file del modello salvato
modello_path = 'models/v6acc80.joblib'

# Carica il modello
modello = joblib.load(modello_path)

df_dati = pd.read_excel('/Users/guglielmo/Desktop/CODE/DataLAB/Aigab/dativ1.5.xlsx', sheet_name='EXAMPLE')

# Effettua le predizioni
predizioni = modello.predict(df_dati)

if predizioni == 1:  
    # Stampa le predizioni
    print("Prodotto consigliabile")
else:
    print("Prodotto non consigliabile")
