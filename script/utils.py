# ===== Importazioni Generali =====
from typing import List, Tuple, Dict, Any, Optional, Callable
import os
import requests
import json
import asyncio
import psycopg2
import socket
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from subprocess import run, CalledProcessError
from itertools import permutations, combinations
from fractions import Fraction
import statistics
import math
import random
import xml.etree.ElementTree as ET
from PIL import Image
import shutil
import hashlib
import logging
from contextlib import contextmanager
import csv
import zipfile
from cryptography.fernet import Fernet
import re
import inquirer  # External library for interactive prompts
import watchdog.observers  # External library for file monitoring
import watchdog.events
import pymongo  # External library for MongoDB interactions


# ========= Lavoro con File =========
# Sezione dedicata alle funzioni per la gestione dei file

def read_file(filename: str) -> str:
    """Legge e ritorna il contenuto di un file testuale."""
    try:
        with managed_file(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "Il file non esiste."

def write_file(filename: str, content: str, mode: str = 'w') -> None:
    """Scrive o appende contenuto in un file."""
    with managed_file(filename, mode) as file:
        file.write(content)

def file_exists(filename: str) -> bool:
    """Controlla l'esistenza di un file."""
    return os.path.exists(filename)

def delete_file(filename: str) -> None:
    """Elimina un file se esiste."""
    try:
        os.remove(filename)
    except FileNotFoundError:
        print("Il file specificato non esiste.")


# ========== API HTTP ==========
# Sezione per funzioni che interagiscono con API HTTP

def fetch_data(url: str, params: Optional[Dict[str, str]] = None) -> Dict:
    """Ottiene dati da un endpoint API usando una richiesta GET."""
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

# =========== Liste ===========
# Manipolazione e operazioni con liste

def create_list(elements: List[Any]) -> List[Any]:
    """Crea e ritorna una lista con gli elementi specificati."""
    return elements

def append_to_list(lst: List[Any], element: Any) -> None:
    """Appende un elemento alla lista."""
    lst.append(element)
    

def remove_duplicates(sequence: List[Any]) -> List[Any]:
    """Remove duplicates from a list while preserving order."""
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

# ========== Dizionari ==========
# Funzioni per la creazione e manipolazione di dizionari

def create_dict(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Crea e ritorna un dizionario dai coppie chiave-valore fornite."""
    return dict(pairs)

# ===== Sistema Operativo =====
# Interazioni con il sistema operativo e il filesystem

def list_directory_contents(directory: str) -> List[str]:
    """Elenca i contenuti di una directory."""
    try:
        return os.listdir(directory)
    except FileNotFoundError:
        return ["La directory non esiste."]

# ========= Database =========
# Connessione e operazioni con database

def connect_to_database(dbname: str, user: str, password: str, host: str) -> psycopg2.extensions.connection:
    """Stabilisce una connessione con un database PostgreSQL."""
    try:
        return psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
    except psycopg2.OperationalError as e:
        print("Impossibile connettersi al database:", str(e))
        return None

# ========= Asincrono =========
# Funzioni per il recupero di dati in modo asincrono

async def fetch_data_async(url: str) -> None:
    """Simula il recupero di dati in modo asincrono."""
    print("Inizio del recupero dati...")
    await asyncio.sleep(2)
    print("Dati recuperati.")

# ========= Networking =========
# Creazione di connessioni socket e comunicazione di rete

def create_socket_connection(host: str, port: int) -> Optional[socket.socket]:
    """Crea e restituisce una connessione socket a un host e porta specificati."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        return s
    except socket.error as e:
        print("Errore nella creazione del socket:", str(e))
        return None

# ====== Visualizzazione Dati ======
# Creazione di grafici e visualizzazioni con Matplotlib

def create_basic_plot(x: List[float], y: List[float], title: str = "Grafico di Base", xlabel: str = "X", ylabel: str = "Y") -> None:
    """Crea e mostra un grafico di base con Matplotlib."""
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# ====== Gestione del Tempo ======
# Funzioni per la gestione delle date e degli orari

def get_current_time() -> datetime:
    """Restituisce l'ora corrente."""
    return datetime.now()

def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Formatta un oggetto datetime in una stringa secondo il formato fornito."""
    return dt.strftime(fmt)

# ====== Lettura e Scrittura JSON ======
def read_json(filename: str) -> Dict:
    """Legge un file JSON e ritorna il suo contenuto come dizionario."""
    try:
        with managed_file(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Errore: Il file {filename} non esiste.")
        return {}

def write_json(filename: str, data: Dict) -> None:
    """Scrive un dizionario in un file JSON."""
    with managed_file(filename, 'w') as f:
        json.dump(data, f, indent=4)

# ====== Lettura e Scrittura XML ======
def read_xml(filename: str) -> ET.Element:
    """Legge un file XML e ritorna la sua radice."""
    try:
        tree = ET.parse(filename)
        return tree.getroot()
    except ET.ParseError:
        print(f"Errore: Il file {filename} non è un XML valido.")
        return None
    except FileNotFoundError:
        print(f"Errore: Il file {filename} non esiste.")
        return None

def write_xml(filename: str, root: ET.Element) -> None:
    """Scrive una struttura XML in un file, partendo dall'elemento radice."""
    tree = ET.ElementTree(root)
    tree.write(filename)

# ========= Manipolazione e Gestione di Immagini =========
def resize_image(input_path: str, output_path: str, size: Tuple[int, int]) -> None:
    """Ridimensiona un'immagine al percorso specificato e la salva in un nuovo file."""
    with Image.managed_file(input_path) as img:
        resized_img = img.resize(size)
        resized_img.save(output_path)

def rotate_image(input_path: str, output_path: str, degrees: float, expand: bool = True) -> None:
    """Ruota un'immagine di un certo numero di gradi e salva il risultato."""
    with Image.managed_file(input_path) as img:
        rotated_img = img.rotate(degrees, expand=expand)
        rotated_img.save(output_path)

# ========= Funzioni Avanzate per la Gestione del Sistema Operativo =========
def copy_file(source: str, destination: str) -> None:
    """Copia un file da un percorso sorgente a un percorso destinazione."""
    try:
        shutil.copy(source, destination)
    except IOError as e:
        print(f"Errore durante la copia del file: {e}")

def move_file(source: str, destination: str) -> None:
    """Sposta un file da un percorso sorgente a un percorso destinazione."""
    try:
        shutil.move(source, destination)
    except IOError as e:
        print(f"Errore durante lo spostamento del file: {e}")

def create_symlink(source: str, link_name: str) -> None:
    """Crea un collegamento simbolico (symlink) a un file o cartella."""
    try:
        os.symlink(source, link_name)
    except OSError as e:
        print(f"Errore durante la creazione del symlink: {e}")

def get_disk_usage(path: str) -> shutil.disk_usage:
    """Ritorna l'uso del disco di un percorso specificato."""
    try:
        return shutil.disk_usage(path)
    except FileNotFoundError:
        print(f"Errore: Il percorso {path} non esiste.")
        return None

# ========= Crittografia e Sicurezza =========
def generate_hash(data: str, algorithm: str = 'sha256') -> str:
    """Genera un hash di una stringa utilizzando l'algoritmo specificato."""
    if hasattr(hashlib, algorithm):
        hasher = getattr(hashlib, algorithm)()
        hasher.update(data.encode())
        return hasher.hexdigest()
    else:
        raise ValueError(f"Algoritmo {algorithm} non supportato.")

# ========= Logging e Monitoraggio =========
def configure_logging(level: str = 'INFO', log_file: str = None) -> None:
    """Configura il logging a livello globale."""
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(filename=log_file, level=level, format=format)
    else:
        logging.basicConfig(level=level, format=format)

# ========= Contesti di Gestione Risorse =========
@contextmanager
def managed_file(filename: str, mode: str = 'r'):
    """Fornisce un contesto per la gestione di un file."""
    try:
        file = managed_file(filename, mode)
        yield file
    finally:
        file.close()


# ========= Gestione delle Variabili d'Ambiente =========
def get_env_variable(key: str, default: Optional[str] = None) -> str:
    """Restituisce il valore di una variabile d'ambiente o un valore di default."""
    return os.getenv(key, default)

# ====== Conversione dei Dati ======
def csv_to_json(csv_filepath: str, json_filepath: str) -> None:
    """Converte un file CSV in un file JSON."""
    data = []
    with managed_file(csv_filepath, encoding='utf-8') as csvf:
        csv_reader = csv.DictReader(csvf)
        for row in csv_reader:
            data.append(row)
    with managed_file(json_filepath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

def xml_to_csv(xml_filepath: str, csv_filepath: str, tag: str) -> None:
    """Converte un file XML in un file CSV, data una specifica etichetta XML."""
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    data = [child.attrib for child in root.findall(tag)]
    keys = data[0].keys()
    with managed_file(csv_filepath, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

# ====== Compressione e Decompressione ======
def compress_files(zip_filepath: str, files: List[str]) -> None:
    """Comprime una lista di file in un archivio ZIP."""
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))

def decompress_zip(zip_filepath: str, extract_dir: str) -> None:
    """Decomprime un archivio ZIP in una directory specificata."""
    with zipfile.ZipFile(zip_filepath, 'r') as zipf:
        zipf.extractall(extract_dir)

# ====== Crittografia e Decrittografia ======
def encrypt_file(file_path: str, key: bytes) -> None:
    """Cripta un file con Fernet."""
    fernet = Fernet(key)
    with managed_file(file_path, 'rb') as file:
        original = file.read()
    encrypted = fernet.encrypt(original)
    with managed_file(file_path, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)

def decrypt_file(file_path: str, key: bytes) -> None:
    """Decripta un file con Fernet."""
    fernet = Fernet(key)
    with managed_file(file_path, 'rb') as file:
        encrypted = file.read()
    decrypted = fernet.decrypt(encrypted)
    with managed_file(file_path, 'wb') as decrypted_file:
        decrypted_file.write(decrypted)

# ====== Pulizia dei Dati ======
def sanitize_input(input_str: str, allow_patterns: List[str]) -> str:
    """Rimuove caratteri non consentiti da una stringa, basandosi su espressioni regolari consentite."""
    for pattern in allow_patterns:
        input_str = re.sub(pattern, '', input_str)
    return input_str

def remove_duplicates_from_file(file_path: str) -> None:
    """Rimuove le righe duplicate da un file testuale."""
    with managed_file(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    unique_lines = list(set(lines))
    with managed_file(file_path, 'w', encoding='utf-8') as file:
        file.writelines(unique_lines)

# ====== Prompt Interattivi ======
def prompt_for_choice(choices: List[str], message: str = "Select an option:") -> str:
    """Mostra un prompt interattivo per scegliere tra diverse opzioni."""
    questions = [
        inquirer.List('choice', message=message, choices=choices),
    ]
    return inquirer.prompt(questions)['choice']

# ====== Monitoraggio dei File ======
class FileChangeHandler(watchdog.events.FileSystemEventHandler):
    def __init__(self, callback: Callable[[str, str], None]):
        self.callback = callback

    def on_modified(self, event):
        if not event.is_directory:
            self.callback(event.src_path, 'modified')

    def on_created(self, event):
        if not event.is_directory:
            self.callback(event.src_path, 'created')

    def on_deleted(self, event):
        if not event.is_directory:
            self.callback(event.src_path, 'deleted')

def start_monitoring_directory(directory: str, callback: Callable[[str, str], None]) -> None:
    """Inizia a monitorare i cambiamenti in una directory, eseguendo un callback per ogni evento."""
    event_handler = FileChangeHandler(callback)
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()

# ====== Persistenza dei Dati ======
def connect_to_mongodb(uri: str, db_name: str) -> pymongo.database.Database:
    """Connette a un database MongoDB e ritorna il riferimento al database."""
    client = pymongo.MongoClient(uri)
    return client[db_name]