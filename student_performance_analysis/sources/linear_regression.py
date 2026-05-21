import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def train_linear_regression(X_train: np.ndarray, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[Modèle] Régression Linéaire entraînée avec succès.")
    return model

def evaluate_model(model: LinearRegression, X_test: np.ndarray, y_test: pd.Series):
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(" --- PERFORMANCES DU MODÈLE ---")
    print(f"R² Score (Précision) : {r2:.4f}")
    print(f"Erreur Moyenne (MAE) : {mae:.2f} points")
    print("---------------------------------\n")

def get_feature_influence(model: LinearRegression, feature_names: list):
    print("--- INFLUENCE DES VARIABLES (COEFFICIENTS) ---")
    # L'interception est la note de départ théorique si toutes les variables valaient 0
    print(f"Note de base (Interception) : {model.intercept_:.2f}")
    
    # Affichage de l'impact de chaque colonne
    for name, coef in zip(feature_names, model.coef_):
        direction = "augmente" if coef > 0 else "diminue"
        print(f"• Chaque hausse de '{name}' {direction} la note de : {abs(coef):.3f} (poids du coefficient)")
    print("-------------------------------------------------\n")