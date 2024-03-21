import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 7. e 8. Creazione di liste di stringhe campione per 'Obb. Finanz.' e 'Esperienza'
obb_finanz = ['accumulo di capitale', 'generazione del reddito', 'conservazione del capitale', 'fondo di emergenza', 'pianificazione della pensione']  
esperienze = ['Principiante', 'Intermedio', 'Avanzato']
features_utenti=['Individuo', 'Dep TOT', 'Q. da inv.', 'Orizz. Temp.', 'Età', 'Deb. In corso',
                            'Avversione al rischio', 'Obb. Finanz.', 'Esperienza', obb_finanz, esperienze]

def creadatiutenti(num_utenti, nomefile, features):
    # Numero di righe dei dati (esclusa l'intestazione)
    num_righe = num_utenti

    # Creazione di un DataFrame vuoto con le colonne specificate
    df = pd.DataFrame(index=range(1, num_righe + 1),
                    columns=features[0:-2])

    # 1. Riempimento della colonna 'Dep TOT' con valori casuali tra 25.000 e 300.000
    df['Dep TOT'] = np.random.randint(25000, 300001, size=num_righe)

    # 2. Riempimento della colonna 'Q. da inv.' con valori calcolati
    df['Q. da inv.'] = df['Dep TOT'] - (df['Dep TOT'] * np.random.rand(num_righe) * 0.95)

    # 3. Riempimento della colonna 'Orizz. Temp.' con valori casuali tra 12 e 120
    df['Orizz. Temp.'] = np.random.randint(12, 121, size=num_righe)

    # 4. Riempimento della colonna 'Età' con una distribuzione normale, arrotondata e limitata tra 22 e 65
    eta_media = (22 + 65) / 2
    eta_std = (65 - 22) / 4
    eta_normale = np.random.normal(eta_media, eta_std, size=num_righe)
    eta_normale = np.clip(eta_normale, 22, 65)
    df['Età'] = np.round(eta_normale)

    # 5. Riempimento della colonna 'Deb. In corso' con valori percentuali casuali tra 0% e 60%
    df['Deb. In corso'] = np.random.randint(0, 61, size=num_righe)

    # 6. Riempimento della colonna 'Avversione al rischio' con valori casuali su una scala da 1 a 10
    df['Avversione al rischio'] = np.random.randint(1, 11, size=num_righe)

    

    # Riempimento delle colonne 'Obb. Finanz.' e 'Esperienza' con valori casuali dalle liste
    df['Obb. Finanz.'] = np.random.choice(features[-2], size=num_righe)
    df['Esperienza'] = np.random.choice(features[-1], size=num_righe)

    # Salvataggio del DataFrame in un file Excel
    print ('OK, dati utenti creati')
    return df.to_excel(f'{nomefile}.xlsx', index=False)

def normalizza_dati_utenti(df, nomefile, features):
    colonne_scaler = features[1:-2]
    scaler = MinMaxScaler()
    df[colonne_scaler] = scaler.fit_transform(df[colonne_scaler])
    
   # Obbiettivi finanziari one-hot encoding
    obb_finanz_cols = features[-2]
    df_obb_finanz = pd.get_dummies(df['Obb. Finanz.'], prefix='Obb')
    # Assicurati che tutte le possibili categorie siano rappresentate
    for col in obb_finanz_cols:
        if f'Obb_{col}' not in df_obb_finanz:
            df_obb_finanz[f'Obb_{col}'] = 0

    # Esperienza di investimento one-hot encoding
    esperienze_cols = features[-1]
    df_esperienze = pd.get_dummies(df['Esperienza'], prefix='Exp')
    # Assicurati che tutte le possibili categorie siano rappresentate
    for col in esperienze_cols:
        if f'Exp_{col}' not in df_esperienze:
            df_esperienze[f'Exp_{col}'] = 0

    # Concatenazione delle nuove colonne al DataFrame originale
    df = pd.concat([df, df_obb_finanz, df_esperienze], axis=1)

    # Rimozione delle colonne originali non più necessarie
    df = df.drop(['Obb. Finanz.', 'Esperienza'], axis=1)
    print ('OK, dati utenti normalizzati')
    return df.to_excel(f'{nomefile}.xlsx', index=False)

# Costanti: elenchi delle caratteristiche dei prodotti
FINALITA_PRODOTTO = ['accumulo di capitale', 'generazione del reddito', 'conservazione del capitale', 'fondo di emergenza', 'pianificazione della pensione']
TIPO_PRODOTTO = ['Azioni', 'Obbligazioni', 'Fondi di investimento', 'Fondi pensione', 'Titoli di stato']
PRESENZA_CEDOLA = ['fissa', 'variabile', 'no']

# Nomi delle colonne del DataFrame
FEATURES_PRODOTTI = ['Nome', 'Scadenza', 'Requisiti minimi di investimento', 'Rendimento atteso', 'Indice di rischio', 'Finalità del prodotto', 'Tipo di prodotto', 'Presenza cedola/dividendo', FINALITA_PRODOTTO, TIPO_PRODOTTO, PRESENZA_CEDOLA]

def crea_prodotti(num_prodotti, nome_file, features):
    """
    Crea un DataFrame di prodotti con dati casuali e salva in un file Excel.

    :param num_prodotti: Numero di prodotti da creare.
    :param nome_file: Nome del file Excel da salvare (senza estensione).
    :return: None
    """
    # Creazione di un DataFrame vuoto con le colonne specificate
    df = pd.DataFrame(index=range(1, num_prodotti + 1), columns=FEATURES_PRODOTTI[0:-3])
    # Popolamento delle colonne con dati casuali
    df['Nome'] = ['Prodotto ' + str(i) for i in range(1, num_prodotti + 1)]
    df['Scadenza'] = np.random.randint(12, 120, size=num_prodotti)
    df['Finalità del prodotto'] = np.random.choice(features[-3], size=num_prodotti)
    df['Tipo di prodotto'] = np.random.choice(features[-2], size=num_prodotti)
    df['Presenza cedola/dividendo'] = np.random.choice(features[-1], size=num_prodotti)
    df['Requisiti minimi di investimento'] = np.random.randint(100, 1000, size=num_prodotti)
    df['Rendimento atteso'] = np.random.uniform(1, 7, size=num_prodotti)  # modificato per usare uniform invece di random
    df['Indice di rischio'] = np.random.randint(1, 7, size=num_prodotti)

    # Salvataggio del DataFrame in un file Excel
    df.to_excel(f'{nome_file}.xlsx', index=False)

def normalizza_dati_prodotti(df, nomefile, features):
    """
    Normalizza i dati dei prodotti e applica la codifica one-hot a categorie specifiche.

    :param df: DataFrame dei prodotti da normalizzare e codificare.
    :param nome_file: Nome del file Excel dove salvare il DataFrame trasformato (senza estensione).
    :return: None
    """
    colonne_scaler = features[1:-6]
    scaler = MinMaxScaler()
    df[colonne_scaler] = scaler.fit_transform(df[colonne_scaler])
    
    # Finalità one-hot encoding
    finalita_cols = features[-3]
    df_fin = pd.get_dummies(df['Finalità del prodotto'], prefix='Fin')
    # Assicurati che tutte le possibili categorie siano rappresentate
    for col in finalita_cols:
        if f'Fin_{col}' not in df_fin:
            df_fin[f'Fin_{col}'] = 0
    
    # Tipologia one-hot encoding
    tipo_cols = features[-2]
    df_tipo = pd.get_dummies(df['Tipo di prodotto'], prefix='Tip')
    # Assicurati che tutte le possibili categorie siano rappresentate
    for col in tipo_cols:
        if f'Tip_{col}' not in df_tipo:
            df_tipo[f'Tip_{col}'] = 0

    # Cedola one-hot encoding
    cedola_cols = features[-1]
    df_cedola = pd.get_dummies(df['Presenza cedola/dividendo'], prefix='Ced')
    # Assicurati che tutte le possibili categorie siano rappresentate
    for col in cedola_cols:
        if f'Ced_{col}' not in df_cedola:
            df_cedola[f'Ced_{col}'] = 0
            
    # Concatenazione delle nuove colonne al DataFrame originale
    df = pd.concat([df,df_fin,df_tipo,df_cedola], axis=1)

    # Rimozione delle colonne originali non più necessarie
    df = df.drop(['Finalità del prodotto', 'Tipo di prodotto', 'Presenza cedola/dividendo'], axis=1)
    print ('OK, dati prodotti normalizzati')
    return df.to_excel(f'{nomefile}.xlsx', index=False)
    
df = pd.read_excel('elenco_prodotti_finanziari.xlsx')
normalizza_dati_prodotti(df, 'prodottinorm', FEATURES_PRODOTTI)

    
