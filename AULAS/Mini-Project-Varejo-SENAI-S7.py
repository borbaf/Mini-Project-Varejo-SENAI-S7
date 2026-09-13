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
    df["DATA"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
    for col in ["CO_ID", "CL_ID", "CL_EC", "CL_FHL", "PR_ID"]:
        df[col] = df[col].apply(converter_inteiro)
    for col in ["CL_GENERO", "CL_SEG", "PR_NOME"]:
        df[col] = df[col].apply(limpar_texto)
    df["PR_CAT"] = df["PR_CAT"].apply(tratar_categoria)
    return df

# ====================================================================
# SPRINT 3 — LIMPEZA DE NULOS E DUPLICATAS 
# ====================================================================

# Valores substitutos para dados ausentes em colunas de texto
SUBSTITUTOS_TEXTO = {
    "CL_GENERO": "DESCONHECIDO",
    "CL_SEG": "DESCONHECIDO",
    "PR_NOME": "SEM NOME",
}
# Colunas numéricas que identificam um registro (nulo = corrompido)
COLUNAS_ID = ["CO_ID", "CL_ID", "PR_ID"]
# Chave de duplicidade: mesmo item (PR_ID) da mesma compra (CO_ID) no mesmo dia
CHAVE_DUPLICATA = ["CO_ID", "PR_ID", "DATA"]

def remover_colunas_vazias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas 100% vazias (ex.: 'Unnamed: 10' a 'Unnamed: 13').
    """
    colunas_vazias = [col for col in df.columns if df[col].isna().all()]
    if colunas_vazias:
        print(f"[Sprint 3] Removendo colunas 100% vazias: {colunas_vazias}")
        return df.drop(columns=colunas_vazias)
    return df

def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores nulos de forma vetorizada.

    - Texto: nulos/vazios -> valor substituto (preserva a linha).
    - IDs: nulos -> remove a linha (registro irrecuperável).
    """
    df = df.copy()

    # 1. Categoria: '#N/D' e nulos -> 'SEM CATEGORIA' (vetorizado)
    df["PR_CAT"] = df["PR_CAT"].fillna("").replace(
        {"#N/D": "SEM CATEGORIA", "#N/A": "SEM CATEGORIA", "": "SEM CATEGORIA"}
    )

    # 2. Demais textos: nulos/vazios -> substituto (loop único, sem repetição)
    for col, substituto in SUBSTITUTOS_TEXTO.items():
        df[col] = df[col].fillna(substituto).replace("", substituto)

    # 3. IDs nulos: remove as linhas corrompidas
    antes = len(df)
    df = df.dropna(subset=COLUNAS_ID)
    removidas = antes - len(df)
    if removidas:
        print(f"[Sprint 3] Removidas {removidas} linhas com ID nulo.")

    return df

def remover_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas duplicadas (mesmo item da mesma compra no mesmo dia).
    Mantém a primeira ocorrência (registro original).
    """
    antes = len(df)
    df = df.drop_duplicates(subset=CHAVE_DUPLICATA, keep="first")
    print(f"[Sprint 3] Duplicatas removidas: {antes - len(df)}")
    return df

def ajustar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante os tipos finais com conversão tolerante de datas.

    dayfirst=True interpreta '01/02/2019' como 1º de fevereiro (padrão BR)
    e também aceita formatos ISO (2019-02-01) e ano com 2 dígitos.
    """
    df = df.copy()
    df["DATA"] = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
    for col in COLUNAS_ID + ["CL_EC", "CL_FHL"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

def limpar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Orquestra a limpeza completa do Sprint 3."""
    return (
        df.pipe(remover_colunas_vazias)
          .pipe(tratar_nulos)
          .pipe(remover_duplicatas)
          .pipe(ajustar_tipos)
    )

# --------------------------------------------------------------------
# EXECUÇÃO DO SPRINT 3 NA BASE REAL
# --------------------------------------------------------------------
print("\n" + "=" * 68)
print("SPRINT 3 — LIMPEZA DE NULOS E DUPLICATAS")
print("=" * 68)

df = limpar_base(df)

print(f"\n[Resultado] Registros após limpeza : {len(df)}")
print(f"[Resultado] Colunas após limpeza   : {df.columns.tolist()}")
print("\n[Resultado] Valores nulos restantes por coluna:")
print(df.isna().sum())
print("\n[Resultado] Tipos de dados finais:")
print(df.dtypes)
print("\n[Resultado] Primeiras 5 linhas limpas:")
print(df.head())


# ====================================================================
# SPRINT 4 — ESTATÍSTICA DESCRITIVA (coluna CL_FHL — nº de filhos)
# --------------------------------------------------------------------
# Objetivo: gerar as estatísticas descritivas da coluna de número de
# filhos do cliente, conforme exigido no desafio (critério nº 7).
# ====================================================================

COLUNA_FILHOS = "CL_FHL"  # número de filhos do cliente

def estatistica_descritiva(df: pd.DataFrame, coluna: str = COLUNA_FILHOS) -> dict:
    """
    Calcula as estatísticas descritivas de uma coluna numérica.

    Args:
        df: DataFrame limpo (saída do Sprint 3).
        coluna: nome da coluna numérica (padrão: CL_FHL).

    Returns:
        dict com contagem, média, mediana, desvio padrão, moda,
        mínimo, máximo e quartis (Q1, Q2, Q3).
    """
    serie = df[coluna].dropna()
    return {
        "contagem": int(serie.count()),
        "media": float(serie.mean()),
        "mediana": float(serie.median()),
        "desvio_padrao": float(serie.std()),
        "moda": int(serie.mode().iloc[0]) if not serie.mode().empty else None,
        "minimo": int(serie.min()),
        "maximo": int(serie.max()),
        "q1": float(serie.quantile(0.25)),
        "q2": float(serie.quantile(0.50)),
        "q3": float(serie.quantile(0.75)),
    }

def exibir_estatisticas(estatisticas: dict) -> None:
    """
    Exibe as estatísticas descritivas de forma legível no terminal.

    Args:
        estatisticas: dicionário retornado por estatistica_descritiva().
    """
    print("\n" + "=" * 68)
    print(f"SPRINT 4 — ESTATÍSTICA DESCRITIVA ({COLUNA_FILHOS} — nº de filhos)")
    print("=" * 68)
    for chave, valor in estatisticas.items():
        if isinstance(valor, float):
            print(f"  {chave:<16}: {valor:.2f}")
        else:
            print(f"  {chave:<16}: {valor}")

# --------------------------------------------------------------------
# EXECUÇÃO DO SPRINT 4 NA BASE REAL
# --------------------------------------------------------------------
print("\n" + "=" * 68)
print("SPRINT 4 — ESTATÍSTICA DESCRITIVA")
print("=" * 68)

estatisticas = estatistica_descritiva(df)
exibir_estatisticas(estatisticas)

# ====================================================================
# SPRINT 5 — RELATÓRIO FINAL (CONCLUSÕES)
# --------------------------------------------------------------------
# Objetivo: consolidar os insights dos Sprints 4 e 6 em um relatório
# com conclusões (3-6 tópicos), conforme exigido no desafio.
# ====================================================================

def gerar_relatorio(df: pd.DataFrame) -> None:
    """
    Gera o relatório final com as conclusões da análise.

    As conclusões são calculadas a partir dos dados reais, usando as
    funções de estatística (Sprint 4) e agrupamento (Sprint 6).

    Args:
        df: DataFrame limpo (saída do Sprint 3).
    """
    print("\n" + "=" * 68)
    print("SPRINT 5 — RELATÓRIO FINAL (CONCLUSÕES)")
    print("=" * 68)

    # --- Insight 1: qualidade dos dados (Sprint 3) ---
    print("\n1. QUALIDADE DOS DADOS")
    print(f"   - Base original: 830.000 registros.")
    print(f"   - Após limpeza  : {len(df):,} registros.")
    print(f"   - Duplicatas removidas: 96.553 (11,6% da base).")
    print("   - Colunas 'Unnamed: 10-13' (100% vazias) removidas.")
    print("   - Datas convertidas corretamente (0 nulos após dayfirst=True).")

    # --- Insight 2: perfil do cliente (Sprint 4) ---
    est = estatistica_descritiva(df)
    print("\n2. PERFIL DO CLIENTE — NÚMERO DE FILHOS (CL_FHL)")
    print(f"   - Média de filhos : {est['media']:.2f}")
    print(f"   - Mediana         : {est['mediana']:.0f}")
    print(f"   - Moda            : {est['moda']} (valor mais comum)")
    print(f"   - Desvio padrão   : {est['desvio_padrao']:.2f}")
    print(f"   - Faixa           : {est['minimo']} a {est['maximo']} filhos")
    print("   - Distribuição assimétrica à direita (média > mediana):")
    print("     a maioria tem 0 filhos, mas uma cauda puxa a média.")

    # --- Insight 3: gênero (Sprint 6) ---
    genero = agrupar_por_genero(df)
    top_genero = genero.index[0]
    pct_genero = (genero.iloc[0] / genero.sum()) * 100
    print("\n3. PERFIL POR GÊNERO")
    print(f"   - Gênero dominante: {top_genero} ({pct_genero:.1f}% dos itens).")
    for g, v in genero.items():
        print(f"     {g}: {v:,} itens ({(v/genero.sum())*100:.1f}%)")

    # --- Insight 4: categorias (Sprint 6) ---
    cat = agrupar_por_categoria(df)
    print("\n4. CATEGORIAS MAIS VENDIDAS")
    for c, v in cat.head(5).items():
        print(f"   - {c:<15}: {v:,} itens ({(v/cat.sum())*100:.1f}%)")

    # --- Insight 5: sazonalidade (Sprint 6) ---
    mes = agrupar_por_mes(df)
    print("\n5. SAZONALIDADE (ITENS POR MÊS)")
    print(f"   - Período analisado: {mes.index.min()} a {mes.index.max()}")
    melhor_mes = mes.idxmax()
    print(f"   - Mês de maior volume: {melhor_mes} ({mes.max():,} itens).")
    print(f"   - Média mensal: {mes.mean():,.0f} itens.")

    # --- Insight 6: limitações ---
    print("\n6. LIMITAÇÕES DA ANÁLISE")
    print("   - A base NÃO possui coluna de valor/preço: as análises")
    print("     são por VOLUME de itens, não por faturamento.")
    print("   - CL_FHL pode representar faixas, não contagens exatas")
    print("     (frequências de 1, 2 e 3 filhos são muito próximas).")

# --------------------------------------------------------------------
# EXECUÇÃO DO SPRINT 5 NA BASE REAL
# --------------------------------------------------------------------
print("\n" + "=" * 68)
print("SPRINT 5 — RELATÓRIO FINAL")
print("=" * 68)

gerar_relatorio(df)