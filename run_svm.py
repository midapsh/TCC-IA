import os
import pandas as pd
import numpy as np
from datetime import time
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from contextlib import contextmanager
from tqdm import tqdm
import joblib

from utilities import standard_hour  # já vem do repositório


# --------------------
# Funções auxiliares
# --------------------


def find_files(base_path, pattern):
    """Busca recursivamente arquivos contendo 'pattern' no nome."""
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(base_path)
        for file in files
        if pattern in file
    ]


def load_and_clean(files):
    """Carrega CSVs do INMET e aplica renomeação + limpeza inicial."""
    df_list = []
    for file in files:
        df = pd.read_csv(file, encoding="ISO-8859-1", sep=";", skiprows=8)
        df = df.rename(
            columns={
                "DATA (YYYY-MM-DD)": "Data",
                "HORA (UTC)": "Hora UTC",
                "RADIACAO GLOBAL (KJ/m²)": "RADIACAO GLOBAL (Kj/m²)",
            }
        )
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True).drop(columns=["Unnamed: 19"])

    # Converter datas
    df["Data"] = pd.to_datetime(df["Data"].str.replace("-", "/"))
    df["Hora UTC"] = pd.to_datetime(
        df["Hora UTC"].apply(standard_hour), format="%H%M"
    ).dt.time

    # Corrigir colunas numéricas (vírgula -> ponto)
    cols_to_float = [
        "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
        "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
        "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)",
        "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)",
        "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)",
        "RADIACAO GLOBAL (Kj/m²)",
        "TEMPERATURA DO PONTO DE ORVALHO (°C)",
        "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)",
        "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)",
        "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)",
        "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)",
        "VENTO, VELOCIDADE HORARIA (m/s)",
    ]
    for col in cols_to_float:
        df[col] = df[col].str.replace(",", ".").astype(float)

    # Rajada de vento já vem como float
    df["VENTO, RAJADA MAXIMA (m/s)"] = df[
        "VENTO, DIREÇÃO HORARIA (gr) (° (gr))"
    ].astype(float)

    return df


def remove_outliers(df):
    """Remove registros sem radiação solar entre 6h e 20h."""
    inicio, fim = time(6, 0), time(20, 0)
    mask = (
        df["RADIACAO GLOBAL (Kj/m²)"].isnull()
        & (df["Hora UTC"] >= inicio)
        & (df["Hora UTC"] < fim)
    )
    return df[~mask]


def engineer_features(df):
    """Cria datetime, year, month, day, hour e remove colunas irrelevantes."""
    df["datetime"] = pd.to_datetime(
        df["Data"].astype(str) + " " + df["Hora UTC"].astype(str)
    )
    df["RADIACAO GLOBAL (Kj/m²)"] = df["RADIACAO GLOBAL (Kj/m²)"].fillna(0)

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour

    drop_cols = [
        "Data",
        "Hora UTC",
        "datetime",
        "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)",
        "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)",
        "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)",
        "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)",
        "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)",
        "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)",
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna()


def split_xy(df, target="TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"):
    x = df.drop(columns=[target])
    y = df[target]
    return train_test_split(x, y, test_size=0.2, random_state=42)


# --------------------
# Progress bar para GridSearch
# --------------------
@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# --------------------
# Treino e salvamento
# --------------------
def train_and_save(x_train, y_train, search_space, filename="best_svr_model.pkl"):
    svr = SVR()
    gs = GridSearchCV(
        estimator=svr,
        param_grid=search_space,
        scoring=["r2", "neg_root_mean_squared_error", "neg_median_absolute_error"],
        refit="r2",
        verbose=2,
        n_jobs=-1,
        cv=3,
    )

    n_candidates = len(list(ParameterGrid(search_space))) * 3  # folds
    with tqdm_joblib(tqdm(total=n_candidates)) as progress:
        gs.fit(x_train, y_train)

    print("Best estimator:", gs.best_estimator_)
    print("Best params:", gs.best_params_)

    joblib.dump(gs.best_estimator_, filename)
    joblib.dump(gs, "gridsearch_svr.pkl")
    return gs


# --------------------
# Main pipeline
# --------------------
def main():
    base_path = "/home/dolores/Documents/matheus-ferreira/TCC-IA/data/_extracted_files"
    pattern = "INMET_SE_SP_A705_BAURU"

    files = find_files(base_path, pattern)
    df = load_and_clean(files)
    df = remove_outliers(df)
    df = engineer_features(df)

    x_train, x_test, y_train, y_test = split_xy(df)

    search_space_svr = {
        "kernel": ["poly"],
        "degree": [2, 3, 4],
        "gamma": ["auto"],
        "tol": [1e-3, 1e-4],
        "C": [0.1, 1],
        "epsilon": [0.01, 0.1],
    }

    train_and_save(x_train, y_train, search_space_svr)


if __name__ == "__main__":
    main()
