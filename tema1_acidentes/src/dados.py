"""Carga, auditoria, limpeza e preparação das features.

Mesmo tratamento do notebook, com uma correção: o dataset tem as datas
deslocadas em 12 anos (o Index começa com o ano real: 2007 -> 2019,
2008 -> 2020, 2009 -> 2021, 2010 -> 2022). O dia da semana é calculado
na data real (data - 12 anos); com a data do dataset ele fica errado em 1 dia.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CATS = ["Light_Conditions", "Road_Surface_Conditions", "Road_Type",
        "Urban_or_Rural_Area", "Weather_Conditions", "Vehicle_Type", "District Area"]
NUM = ["Latitude", "Longitude", "month_sin", "month_cos",
       "dow_sin", "dow_cos", "is_weekend"]
DESLOCAMENTO_ANOS = 12


def carregar(caminho_csv):
    df = pd.read_csv(caminho_csv, low_memory=False)
    df["Accident Date"] = pd.to_datetime(df["Accident Date"], errors="coerce", dayfirst=True)
    return df


def limpar(df):
    """Remove Index, duplicatas, o bloco 2023-2025 e linhas sem coordenadas."""
    d = df.drop(columns=["Index"]).drop_duplicates()
    d = d[d["Accident Date"] < "2023-01-01"].copy()
    d = d.dropna(subset=["Latitude", "Longitude"])
    for c in CATS:
        d[c] = d[c].fillna("Unknown")
    d["Urban_or_Rural_Area"] = d["Urban_or_Rural_Area"].replace("Unallocated", "Unknown")
    d["y"] = d["Accident_Severity"].map({"Slight": 0, "Serious": 1, "Fatal": 2}).astype("int8")
    d["grave"] = (d["y"] >= 1).astype("int8")
    # data real do acidente (o dataset somou 12 anos ao ano original)
    d["data_real"] = d["Accident Date"] - pd.DateOffset(years=DESLOCAMENTO_ANOS)
    return d.reset_index(drop=True)


def features_calendario(d, corrigir_dia_semana=True):
    base = d["data_real"] if corrigir_dia_semana else d["Accident Date"]
    mes, dow = d["Accident Date"].dt.month, base.dt.dayofweek
    d = d.copy()
    d["month_sin"] = np.sin(2 * np.pi * mes / 12)
    d["month_cos"] = np.cos(2 * np.pi * mes / 12)
    d["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    d["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    d["is_weekend"] = (dow >= 5).astype("int8")
    return d


def dividir_aleatorio(d, semente=42):
    """70/15/15 estratificado pela gravidade (divisão principal)."""
    train, temp = train_test_split(d, test_size=0.30, stratify=d["y"], random_state=semente)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["y"], random_state=semente)
    return train, val, test


def dividir_temporal(d, semente=42):
    """Treina nos 3 primeiros anos e testa no último (verificação extra).

    Treino/validação: 2007-2009 reais (2019-2021 no dataset), validação = 15% sorteado.
    Teste: 2010 real (2022 no dataset).
    """
    passado = d[d["Accident Date"].dt.year <= 2021]
    futuro = d[d["Accident Date"].dt.year == 2022]
    train, val = train_test_split(passado, test_size=0.15, stratify=passado["y"],
                                  random_state=semente)
    return train, val, futuro


class Preparador:
    """Vocabulários, média e desvio aprendidos só no treino. Código 0 = categoria nova."""

    def __init__(self, num=NUM):
        self.num = list(num)

    def ajustar(self, train):
        self.maps = {c: {v: i + 1 for i, v in enumerate(sorted(train[c].unique()))} for c in CATS}
        self.mu = train[["Latitude", "Longitude"]].mean().to_dict()
        self.sd = train[["Latitude", "Longitude"]].std().to_dict()
        if "Number_of_Vehicles" in self.num:
            self.mu["Number_of_Vehicles"] = float(train["Number_of_Vehicles"].mean())
            self.sd["Number_of_Vehicles"] = float(train["Number_of_Vehicles"].std())
        return self

    @property
    def vocab(self):
        return [len(self.maps[c]) + 1 for c in CATS]

    def transformar(self, part):
        X_cat = np.stack([part[c].map(self.maps[c]).fillna(0).astype("int64").to_numpy()
                          for c in CATS], axis=1)
        X_num = part[self.num].astype("float64").copy()
        for c in self.mu:
            if c in X_num:
                X_num[c] = (X_num[c].fillna(self.mu[c]) - self.mu[c]) / self.sd[c]
        return X_cat, X_num.to_numpy("float32"), part["grave"].to_numpy()
