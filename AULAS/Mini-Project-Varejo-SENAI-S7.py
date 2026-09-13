"""
Sprint 1 — Importação dos dados
Baixa a base Varejo do Kaggle, descompacta e carrega em um DataFrame.
"""
import os
import re
import zipfile
from datetime import datetime
from typing import Optional
import unicodedata

import kagglehub
import pandas as pd

def baixar_base_varejo() -> str:
    """Baixa o dataset do Kaggle e retorna o caminho do CSV real (descompactado)."""
    # 1. Baixa o dataset (kagglehub cacheia localmente)
    path = kagglehub.dataset_download("namespaiva/base-varejo")
    print(f"✅ Dataset baixado em: {path}")

    # 2. Localiza o arquivo baixado (o "Base Varejo.csv" do Kaggle é um ZIP)
    arquivo = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                arquivo = os.path.join(root, f)
                break
        if arquivo:
            break

    # 3. Verifica se é um ZIP pelos bytes mágicos (PK\x03\x04)
    with open(arquivo, "rb") as f:
        is_zip = f.read(4) == b"PK\x03\x04"

    if is_zip:
        print("📦 Arquivo é um ZIP. Extraindo...")
        destino = os.path.join(os.path.dirname(arquivo), "extraido")
        os.makedirs(destino, exist_ok=True)
        with zipfile.ZipFile(arquivo, "r") as z:
            z.extractall(destino)
            print("Conteúdo do ZIP:", z.namelist())
            csv_real = os.path.join(destino, z.namelist()[0])
    else:
        csv_real = arquivo

    print(f"✅ CSV pronto: {csv_real}")
    return csv_real

def detectar_separador(caminho: str) -> str:
    """Detecta o separador do CSV analisando a primeira linha."""
    with open(caminho, "r", encoding="latin1") as f:
        linha = f.readline()
    # Conta qual delimitador aparece mais vezes
    delimitadores = {";": linha.count(";"), ",": linha.count(","), "\t": linha.count("\t")}
    return max(delimitadores, key=delimitadores.get)

if __name__ == "__main__":
    csv_path = baixar_base_varejo()
    sep = detectar_separador(csv_path)

    df = pd.read_csv(csv_path, encoding="latin1", sep=sep)

# ====================================================================
# SPRINT 2 — TRANSFORMAÇÃO DE STRINGS, INT, FLOAT E DATETIME
# ====================================================================

CATEGORIAS_INVALIDAS = {"#N/D", "#N/A", "N/D", "N/A", "NA", ""}

def limpar_texto(texto: str) -> str:
    """Remove espaços extras e padroniza para maiúsculas."""
    if not isinstance(texto, str):
        return ""
    return " ".join(texto.split()).upper()

def remover_acentos(texto: str) -> str:
    """Remove acentos via normalização Unicode (NFD)."""
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")

def tratar_categoria(categoria: str) -> str:
    """Substitui categorias ausentes/inválidas por 'SEM CATEGORIA'."""
    if not isinstance(categoria, str):
        return "SEM CATEGORIA"
    categoria = categoria.strip().upper()
    return "SEM CATEGORIA" if categoria in CATEGORIAS_INVALIDAS else categoria

def converter_inteiro(valor: object) -> Optional[int]:
    """Converte para int de forma segura (trata milhar e vírgula decimal)."""
    if isinstance(valor, bool) or valor is None:
        return None
    if isinstance(valor, (int, float)):
        return int(valor)
    if isinstance(valor, str):
        limpo = valor.replace(".", "").replace(",", ".")
        try:
            return int(float(limpo))
        except ValueError:
            return None
    return None

def converter_decimal(valor: object) -> Optional[float]:
    """Converte para float de forma segura (padrão BR: '12,50')."""
    if isinstance(valor, bool) or valor is None:
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    if isinstance(valor, str):
        limpo = valor.replace(".", "").replace(",", ".")
        try:
            return float(limpo)
        except ValueError:
            return None
    return None

def converter_data(data_str: str, formato: str = "%d/%m/%Y") -> Optional[pd.Timestamp]:
    """Converte string de data para datetime; retorna None se inválida."""
    if not isinstance(data_str, str) or not data_str.strip():
        return None
    try:
        return pd.to_datetime(data_str, format=formato, errors="raise")
    except (ValueError, TypeError):
        return None

def validar_data(data_str: str, formato: str = "%d/%m/%Y") -> bool:
    """Valida data estritamente (rejeita 31/02/2019)."""
    if not isinstance(data_str, str):
        return False
    try:
        datetime.strptime(data_str, formato)
        return True
    except ValueError:
        return False

def aplicar_transformacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as transformações de tipo/limpeza no DataFrame."""
    df = df.copy()
    df["DATA"] = pd.to_datetime(df["DATA"], format="%d/%m/%Y", errors="coerce")
    for col in ["CO_ID", "CL_ID", "CL_EC", "CL_FHL", "PR_ID"]:
        df[col] = df[col].apply(converter_inteiro)
    for col in ["CL_GENERO", "CL_SEG", "PR_NOME"]:
        df[col] = df[col].apply(limpar_texto)
    df["PR_CAT"] = df["PR_CAT"].apply(tratar_categoria)
    return df
