import joblib
import pandas as pd
from dati import normalizza_dati_prodotti, normalizza_dati_utenti
# Percorso al file del modello salvato
modello_path = 'models/v6acc80.joblib'

# Carica il modello
modello = joblib.load(modello_path)

OBB_FINANZ = ['accumulo di capitale', 'generazione del reddito', 'conservazione del capitale', 'fondo di emergenza', 'pianificazione della pensione']  
ESPERIENZE = ['Principiante', 'Intermedio', 'Avanzato']
FEATURES_UTENTI=['Individuo',
                 'Dep TOT', 
                 'Q. da inv.', 
                 'Orizz. Temp.', 
                 'Età', 
                 'Deb. In corso',
                 'Avversione al rischio', 
                 'Obb. Finanz.', 
                 'Esperienza', 
                 OBB_FINANZ, 
                 ESPERIENZE]

# Costanti: elenchi delle caratteristiche dei prodotti
FINALITA_PRODOTTO = ['accumulo di capitale', 'generazione del reddito', 'conservazione del capitale', 'fondo di emergenza', 'pianificazione della pensione']
TIPO_PRODOTTO = ['Azioni', 'Obbligazioni', 'Fondi di investimento', 'Fondi pensione', 'Titoli di stato']
PRESENZA_CEDOLA = ['fissa', 'variabile', 'no']

# Nomi delle colonne del DataFrame
FEATURES_PRODOTTI = ['Nome',
                     'Scadenza',
                     'Requisiti minimi di investimento',
                     'Rendimento atteso',
                     'Indice di rischio',
                     'Finalità del prodotto',
                     'Tipo di prodotto',
                     'Presenza cedola/dividendo',
                     FINALITA_PRODOTTO,
                     TIPO_PRODOTTO,
                     PRESENZA_CEDOLA]

dati_utente = {
    'Individuo': 'Gianfranco',
    'Dep TOT' : [50000],
    'Q. da inv.': [25000],
    'Orizz. Temp.': [120],
    'Età': [34],
    'Deb. In corso': [30],
    'Avversione al rischio': [3],
    'Obb. Finanz.': ['accumulo di capitale'],                                           
    'Esperienza': ['Avanzato']
}


dati_prodotto = {
    'Nome': 'Inculata s.p.a.',
    'Scadenza': [100],
    'Requisiti minimi di investimento': [300],
    'Rendimento atteso': [6],
    'Indice di rischio': [4],
    'Finalità del prodotto': ['accumulo di capitale'],
    'Tipo di prodotto': ['Azioni'],
    'Presenza cedola/dividendo': ['fissa']
    
}

df_dati_utente = pd.DataFrame(dati_utente)
print(df_dati_utente)
df_dati_prodotto = pd.DataFrame(dati_prodotto)
print(df_dati_prodotto)

df_dati_utente_norm = normalizza_dati_utenti(df_dati_utente, 'test11', FEATURES_UTENTI)
print(df_dati_utente_norm)

df_dati_prodotto_norm = normalizza_dati_prodotti(df_dati_prodotto, 'test21', FEATURES_PRODOTTI)
print(df_dati_prodotto_norm)

df_dati = pd.concat([df_dati_utente_norm, df_dati_prodotto_norm], axis=1)
print(df_dati)

# Effettua le predizioni
predizioni = modello.predict(df_dati)

# Stampa le predizioni
print("Le predizioni sul nuovo dataset sono:", predizioni)
