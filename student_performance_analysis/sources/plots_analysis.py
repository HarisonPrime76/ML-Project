import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def generate_performance_dashboard(
    X_test_df: pd.DataFrame, 
    y_test: pd.Series, 
    model_lr: LinearRegression, 
    model_rf: RandomForestRegressor,
    feature_names: list
):

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Tableau de Bord - Analyse des Performances des Élèves", fontsize=20, fontweight='bold', y=0.98)


def save_analysis_dashboard(X_test_scaled, y_test, model_lr, model_rf, feature_names, filename):
   
    sns.set_theme(style="whitegrid")
    
    # Création de la matrice de graphiques 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("DASHBOARD TECHNIQUE & PERFORMANCES", fontsize=18, fontweight='bold', color='#2c3e50')

    y_pred_lr = model_lr.predict(X_test_scaled)
    y_pred_rf = model_rf.predict(X_test_scaled)

    # --- GRAPH_1 : Comparaison Réel vs Prédit (Régression Linéaire) ---
    sns.scatterplot(x=y_test, y=y_pred_lr, ax=axes[0, 0], alpha=0.7, color='#3498db', edgecolor='w')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='#e74c3c')
    axes[0, 0].set_title("Régression Linéaire : Réel vs Prédit", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Vrais Scores (total_score)")
    axes[0, 0].set_ylabel("Scores Prédits")

    # --- GRAPH_2 : Comparaison Réel vs Prédit (Random Forest) ---
    sns.scatterplot(x=y_test, y=y_pred_rf, ax=axes[0, 1], alpha=0.7, color='#2ecc71', edgecolor='w')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='#e74c3c')
    axes[0, 1].set_title("Random Forest : Réel vs Prédit", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Vrais Scores (total_score)")
    axes[0, 1].set_ylabel("Scores Prédits")

    # --- GRAPH_3 : Importance des Variables (Coefficients de la Régression) ---
    lr_coefs = np.abs(model_lr.coef_)
    # Normalisation en % pour l'affichage visuel comparatif
    lr_coefs_pct = (lr_coefs / np.sum(lr_coefs)) * 100
    df_lr = pd.DataFrame({'Variable': feature_names, 'Importance': lr_coefs_pct}).sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Variable', data=df_lr, ax=axes[1, 0], palette="Blues_r", hue='Variable', legend=False)
    axes[1, 0].set_title(" Répartition de l'influence (Régression Linéaire %)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Poids relatif (%)")
    axes[1, 0].set_ylabel("")

    # --- GRAPH_4 : Importance des Variables (Random Forest) ---
    rf_importances = model_rf.feature_importances_ * 100
    df_rf = pd.DataFrame({'Variable': feature_names, 'Importance': rf_importances}).sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Variable', data=df_rf, ax=axes[1, 1], palette="Greens_r", hue='Variable', legend=False)
    axes[1, 1].set_title("Importance des Variables (Random Forest %)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Contribution (%)")
    axes[1, 1].set_ylabel("")

    # Ajustement automatique des espacements
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Sauvegarde de l'image
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()
    print(f"[Visualisation] Dashboard de performance sauvegardé sous '{filename}' !")
