from sources import *

def main():
    df = data_load("datasets/student_data.csv")
    df_clean = DataClean(tolerance=1.0)
    data_clean = df_clean.fit_transformer(df)

    X, y = features_xy(data_clean)
    X_train, X_test, y_train, y_test = train_test_process(X, y)

    # PREPROCESSOR
    preprocessor = create_preprocessing_pipeline(X_train)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    print("Données prêtes pour l'entraînement du modèle !")


    #  RÉGRESSION LINÉAIRE
    print("\n--- [MODÈLE 1 : RÉGRESSION LINÉAIRE] ---")
    model_lr = train_linear_regression(X_train_scaled, y_train)
    evaluate_model(model_lr, X_test_scaled, y_test)
    get_feature_influence(model_lr, feature_names=list(X.columns))
    
    # RANDOM FOREST
    print("\n--- [MODÈLE 2 : RANDOM FOREST] ---")
    model_rf = train_random_forest(X_train_scaled, y_train, random_state=RANDOM_STATE)
    evaluate_rf_model(model_rf, X_test_scaled, y_test)
    get_rf_feature_importance(model_rf, feature_names=list(X.columns))

    # PERFROMANCE VISUALISER
    save_analysis_dashboard(
        X_test_scaled=X_test_scaled, 
        y_test=y_test, 
        model_lr=model_lr, 
        model_rf=model_rf, 
        feature_names=list(X.columns), 
        filename="dashboard_performances.png"
    )
    
if __name__ == "__main__":
    main()
    