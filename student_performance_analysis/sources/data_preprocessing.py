import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def create_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
   
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"[Preprocessing] Colonnes numériques détectées : {numeric_features}")
    print(f"[Preprocessing] Colonnes catégorielles détectées : {categorical_features}")


    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Remplace les nans par la médiane
        ('scaler', StandardScaler())                    # Centre et réduit (moyenne=0, variance=1)
    ])

    # 3. Pipeline pour les variables catégorielles (Imputation + Encodage en colonnes 0/1)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Remplace les nans par le plus fréquent
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Crée les variables d'encodage
    ])

    # 4. Assemblage final
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Supprime les colonnes non spécifiées par sécurité
    )

    return preprocessor