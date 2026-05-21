import pandas as pd

def data_load(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("Erreur : Fichier introuvable")
        return 0
    