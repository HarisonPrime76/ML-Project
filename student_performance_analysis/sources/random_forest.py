import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


def train_random_forest(X_train: np.ndarray, y_train: pd.Series, random_state: int) -> RandomForestRegressor:

    # n_estimators=100 signifie que le modèle va créer 100 arbres de décision différents
    # max_depth=5 permet d'éviter que le modèle n'apprenne par cœur (overfitting)
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    model.fit(X_train, y_train)
    print("[Modèle] Random Forest entraîné avec succès.")
    return model

def evaluate_rf_model(model: RandomForestRegressor, X_test: np.ndarray, y_test: pd.Series):

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n --- PERFORMANCES DU RANDOM FOREST ---")
    print(f"R² Score (Précision) : {r2:.4f}")
    print(f"Erreur Moyenne (MAE) : {mae:.2f} points")
    print("----------------------------------------\n")

def get_rf_feature_importance(model: RandomForestRegressor, feature_names: list):
    """
    Affiche l'importance relative de chaque variable (la somme des importances vaut 1.0 ou 100%).
    """
    print("--- IMPORTANCE DES VARIABLES (RANDOM FOREST) ---")
    
    # Récupération des scores d'importance du modèle
    importances = model.feature_importances_
    
    # Tri des variables de la plus importante à la moins importante
    indices = np.argsort(importances)[::-1]
    
    for i in indices:
        print(f"• {feature_names[i]} : {importances[i]*100:.2f}% de contribution à la décision")
    print("---------------------------------------------------\n")