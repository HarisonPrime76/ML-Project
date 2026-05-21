import pandas as pd


class DataClean:

    def __init__(self, tolerance: float = 1.0):
       
        self.tolerance = tolerance
        self.keeping_column: list[str] = []

    def fit(self, df: pd.DataFrame):
       
        nb_lignes = len(df)
        min_require_value = int(self.tolerance * nb_lignes)

       
        df_temp = df.dropna(axis=1, thresh=min_require_value)
        self.keeping_column = df_temp.columns.tolist()

        print(f"[Fit] Colonnes conservées ({len(self.keeping_column)}/{df.shape[1]})")

    def transformer(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self.keeping_column:
            raise ValueError()

        df_clean = df.copy()

        # 2. Clear all duplicate
        nb_dupicate = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        nb_duplicate_remove = nb_dupicate - len(df_clean)

        colonnes_existantes = [col for col in self.keeping_column if col in df_clean.columns]
        df_clean = df_clean[colonnes_existantes]

        print(f" [Transformer] {nb_duplicate_remove} doublon(s) supprimé(s). Dataset final : {df_clean.shape}")
        return df_clean

    def fit_transformer(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transformer(df)