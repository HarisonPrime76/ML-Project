import numpy as np
import pandas as pd

# Pour avoir exactement les mêmes résultats à chaque génération
np.random.seed(42)
n_students = 1000000

# 1. Génération de variables explicatives réalistes
# Heures d'étude : entre 1h et 25h par semaine (moyenne autour de 10h)
weekly_self_study_hours = np.random.gamma(shape=3, scale=3, size=n_students) + 1
weekly_self_study_hours = np.clip(weekly_self_study_hours, 1, 25).round(1)

# Pourcentage de présence : entre 60% et 100% (beaucoup d'élèves assidus)
attendance_percentage = np.random.beta(a=7, b=1.5, size=n_students) * 100
attendance_percentage = attendance_percentage.round(1)

# Participation : note de 1 à 10
class_participation = np.random.randint(1, 11, size=n_students)

# 2. Création de la formule magique cachée pour le 'total_score' (sur 100 points)
# Note de base = 30 points
# Chaque heure d'étude = +1.5 point
# Chaque % de présence = +0.4 point
# Chaque point de participation = +1.2 point
# + Un bruit aléatoire (le facteur chance à l'examen)
base_score = 30
noise = np.random.normal(loc=0, scale=4, size=n_students) # Écart de +/- 4 points

total_score = (
    base_score 
    + (weekly_self_study_hours * 1.5) 
    + ((attendance_percentage - 60) * 0.4) # On donne des points au-dessus de 60%
    + (class_participation * 1.2) 
    + noise
)

total_score = np.clip(total_score, 0, 100).round(1)

df = pd.DataFrame({
    'weekly_self_study_hours': weekly_self_study_hours,
    'attendance_percentage': attendance_percentage,
    'class_participation': class_participation,
    'total_score': total_score
})

df.to_csv('datasets/student_data.csv', index=False)
print("Fichier 'student_data.csv' généré avec succès avec 1000000 lignes !")