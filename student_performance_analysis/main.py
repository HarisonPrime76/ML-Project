
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('dataset/ultimate_student_productivity_dataset_5000.csv')
df = df[df['gender'] != 'Other']
#print(df.head(10))


y = df['exam_score']
X = df.drop(columns=['exam_score', 'student_id'])

num_features = [
    'age', 'study_hours', 'self_study_hours', 'online_classes_hours', 
    'social_media_hours', 'gaming_hours', 'sleep_hours', 'screen_time_hours', 
    'exercise_minutes', 'caffeine_intake_mg', 'mental_health_score', 
    'focus_index', 'burnout_level', 'productivity_score'
]


cat_features = ['gender', 'academic_level', 'internet_quality']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ],
    remainder='passthrough' # Garde les colonnes binaires (0/1) telles quelles
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print("X_train :", {X_train_transformed.shape})


# 1. Création et entraînement du modèle
# On utilise 100 arbres de décision pour la forêt
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_transformed, y_train)

# 2. Prédiction et Évaluation
predictions = model.predict(X_test_transformed)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Erreur moyenne (MAE) : {mae:.2f} points")
print(f"Précision (R²) : {r2:.2%}")

# 3. Extraction de l'importance des variables
# On récupère les noms des colonnes après transformation
encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
feature_names = num_features + list(encoded_cat_names) + ['part_time_job', 'upcoming_deadline']

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# 4. Affichage des 10 facteurs les plus influents
print("\n--- Top 10 des facteurs influençant la note ---")
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.2%})")

# Optionnel : Petit graphique pour visualiser
plt.figure(figsize=(10,6))
plt.title("Importance des variables sur le résultat d'examen")
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
plt.tight_layout()
plt.show()

