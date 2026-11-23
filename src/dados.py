"""Carga, auditoria, limpeza e preparação das features.

Mesmo tratamento do notebook, com duas extensões:

1. O dataset tem as datas deslocadas em 12 anos (o Index começa com o ano real:
   2007 -> 2019, 2008 -> 2020, 2009 -> 2021, 2010 -> 2022). O dia da semana é
   calculado na data real (data - 12 anos).
2. Junção com os dados oficiais completos do Reino Unido (STATS19, arquivo
   Accidents0515.csv), que trazem limite de velocidade, horário, tipo e controle
   de cruzamento, classe da via etc. Só entram variáveis do local e do momento,
   conhecidas antes do acidente (nada de vítimas, veículos ou atendimento policial).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CATS = ["Light_Conditions", "Road_Surface_Conditions", "Road_Type",
        "Urban_or_Rural_Area", "Weather_Conditions", "Vehicle_Type", "District Area"]
NUM = ["Latitude", "Longitude", "month_sin", "month_cos",
       "dow_sin", "dow_cos", "is_weekend"]
DESLOCAMENTO_ANOS = 12

# variáveis vindas dos dados oficiais (códigos do STATS19, tratados como categorias)
CATS_OFICIAIS = ["Speed_limit", "Junction_Detail", "Junction_Control", "1st_Road_Class",
                 "2nd_Road_Class", "Pedestrian_Crossing-Human_Control",
                 "Pedestrian_Crossing-Physical_Facilities", "Special_Conditions_at_Site",
                 "Carriageway_Hazards"]
NUM_OFICIAIS = ["speed", "hour_sin", "hour_cos", "hora_ok"]
PADRONIZAR = ["Latitude", "Longitude", "speed", "Number_of_Vehicles"]


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


def _chave(df, data, sev):
    """Chave sem o Index (que está corrompido em ~36% das linhas)."""
    return (df[data].dt.strftime("%Y%m%d") + "|" + df["Latitude"].round(5).astype(str) + "|" +
            df["Longitude"].round(5).astype(str) + "|" + df[sev].astype(str) + "|" +
            df["Number_of_Vehicles"].astype(str) + "|" + df["Number_of_Casualties"].astype(str))


def juntar_oficial(d, caminho_stats19):
    """Acrescenta as variáveis dos dados oficiais (STATS19 2007-2010).

    A junção usa data real + coordenadas + gravidade + nº de veículos e de vítimas
    (as duas últimas só servem para casar as linhas; não entram no modelo).
    Casa 99,97% das linhas; o resto fica com código "-9" (desconhecido).
    """
    s = pd.read_csv(caminho_stats19, low_memory=False, encoding="utf-8-sig")
    s = s[s["Accident_Index"].astype(str).str[:4].isin(["2007", "2008", "2009", "2010"])].copy()
    s["data_real"] = pd.to_datetime(s["Date"], dayfirst=True)
    s["sev_txt"] = s["Accident_Severity"].map({1: "Fatal", 2: "Serious", 3: "Slight"})
    s["k"] = _chave(s, "data_real", "sev_txt")
    s = s[~s["k"].duplicated(keep=False)]
    d = d.copy()
    d["k"] = _chave(d, "data_real", "Accident_Severity")
    out = d.merge(s[["k", "Time"] + CATS_OFICIAIS], on="k", how="left").drop(columns="k")
    assert len(out) == len(d)
    out["casou_oficial"] = out["Speed_limit"].notna().astype("int8")
    t = pd.to_datetime(out["Time"], format="%H:%M", errors="coerce")
    hora = t.dt.hour + t.dt.minute / 60
    out["hora_ok"] = hora.notna().astype("int8")
    hora = hora.fillna(12)
    out["hour_sin"] = np.sin(2 * np.pi * hora / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hora / 24)
    out["speed"] = out["Speed_limit"].fillna(30).astype(float)
    for c in CATS_OFICIAIS:
        out[c] = out[c].fillna(-9).astype(int).astype(str)
    return out


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

    def __init__(self, num=NUM, cats=CATS):
        self.num = list(num)
        self.cats = list(cats)

    def ajustar(self, train):
        self.maps = {c: {v: i + 1 for i, v in enumerate(sorted(train[c].unique()))} for c in self.cats}
        self.mu, self.sd = {}, {}
        for c in PADRONIZAR:
            if c in self.num:
                self.mu[c] = float(train[c].mean())
                self.sd[c] = float(train[c].std())
        return self

    @property
    def vocab(self):
        return [len(self.maps[c]) + 1 for c in self.cats]

    def transformar(self, part):
        X_cat = np.stack([part[c].map(self.maps[c]).fillna(0).astype("int64").to_numpy()
                          for c in self.cats], axis=1)
        X_num = part[self.num].astype("float64").copy()
        for c in self.mu:
            X_num[c] = (X_num[c].fillna(self.mu[c]) - self.mu[c]) / self.sd[c]
        return X_cat, X_num.to_numpy("float32"), part["grave"].to_numpy()
