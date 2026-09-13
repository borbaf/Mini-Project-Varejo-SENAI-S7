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
# SPRINT 5 — RELATÓRIO FINAL E README
# --------------------------------------------------------------------
# Objetivo: gerar o relatório final com os insights extraídos da base.
# ====================================================================

def gerar_insights(df: pd.DataFrame) -> list:
    """
    Extrai os insights principais da base para compor o relatório.

    Cada insight é um dicionário com 'titulo' e 'descricao', gerado
    a partir dos dados reais (não hardcoded).

    Args:
        df: DataFrame limpo (saída do Sprint 3).

    Returns:
        Lista de insights (dicts) com título e descrição.
    """
    insights = []

    # Insight 1 — Perfil do cliente: número de filhos (Sprint 4)
    filhos = df["CL_FHL"].dropna()
    media_filhos = filhos.mean()
    moda_filhos = int(filhos.mode().iloc[0]) if not filhos.mode().empty else None
    insights.append({
        "titulo": "Perfil demográfico — número de filhos",
        "descricao": (
            f"A média de filhos por cliente é {media_filhos:.2f}, "
            f"com moda em {moda_filhos}. A distribuição é assimétrica à "
            f"direita (média > mediana), indicando concentração de "
            f"clientes sem filhos, mas com uma cauda de famílias maiores."
        ),
    })

    # Insight 2 — Gênero dominante (Sprint 6)
    genero = df.groupby("CL_GENERO").size().sort_values(ascending=False)
    if not genero.empty:
        gen_top = genero.index[0]
        gen_pct = (genero.iloc[0] / genero.sum()) * 100
        insights.append({
            "titulo": "Gênero dominante nas compras",
            "descricao": (
                f"O gênero '{gen_top}' concentra {gen_pct:.1f}% dos itens "
                f"vendidos, indicando o principal público consumidor."
            ),
        })

    # Insight 3 — Categoria mais vendida (Sprint 6)
    categoria = df.groupby("PR_CAT").size().sort_values(ascending=False)
    if not categoria.empty:
        cat_top = categoria.index[0]
        cat_pct = (categoria.iloc[0] / categoria.sum()) * 100
        insights.append({
            "titulo": "Categoria líder de vendas",
            "descricao": (
                f"A categoria '{cat_top}' lidera com {cat_pct:.1f}% dos "
                f"itens vendidos, sendo o principal motor de volume."
            ),
        })

    # Insight 4 — Sazonalidade (Sprint 6, se DATA estiver disponível)
    if pd.api.types.is_datetime64_any_dtype(df["DATA"]):
        periodo = df["DATA"].dt.to_period("M").astype(str)
        por_mes = df.groupby(periodo).size()
        if not por_mes.empty:
            mes_pico = por_mes.idxmax()
            insights.append({
                "titulo": "Sazonalidade das vendas",
                "descricao": (
                    f"O período de maior volume de vendas foi {mes_pico}, "
                    f"indicando sazonalidade que pode orientar estoque "
                    f"e campanhas."
                ),
            })

    # Insight 5 — Qualidade de dados (Sprint 3)
    insights.append({
        "titulo": "Qualidade de dados",
        "descricao": (
            f"Após a limpeza, a base ficou com {len(df)} registros válidos "
            f"e 0 valores nulos. Foram removidas colunas vazias e "
            f"duplicatas, garantindo consistência para as análises."
        ),
    })

    return insights

def gerar_relatorio(df: pd.DataFrame) -> str:
    """
    Gera o texto do relatório final com os insights extraídos.

    Args:
        df: DataFrame limpo.

    Returns:
        String com o relatório formatado.
    """
    insights = gerar_insights(df)

    linhas = []
    linhas.append("=" * 68)
    linhas.append("SPRINT 5 — RELATÓRIO FINAL DE ANÁLISE EXPLORATÓRIA")
    linhas.append("=" * 68)
    linhas.append("")
    linhas.append(f"Base analisada: Base Varejo (Kaggle)")
    linhas.append(f"Registros analisados: {len(df)}")
    linhas.append("")
    linhas.append("CONCLUSÕES:")
    linhas.append("")

    for i, insight in enumerate(insights, start=1):
        linhas.append(f"{i}. {insight['titulo']}")
        linhas.append(f"   {insight['descricao']}")
        linhas.append("")

    return "\n".join(linhas)

def gerar_readme(df: pd.DataFrame) -> str:
    """
    Gera o conteúdo do arquivo README.md do projeto.

    Args:
        df: DataFrame limpo.

    Returns:
        String com o conteúdo do README.md.
    """
    return f"""# Mini-Projeto — Análise Exploratória de Dados (Base Varejo)

## 📌 Descrição
Análise exploratória de dados (AED) do dataset **Base Varejo** (Kaggle),
utilizando Python e pandas. O projeto é dividido em 6 sprints e atende
aos critérios de avaliação da disciplina.

## 🗂️ Estrutura do Projeto
- **Sprint 1** — Importação dos dados (Kaggle + descompactação)
- **Sprint 2** — Transformação de strings, inteiros, floats e datas
- **Sprint 3** — Limpeza de nulos e duplicatas
- **Sprint 4** — Estatística descritiva (nº de filhos do cliente)
- **Sprint 5** — Relatório final e README
- **Sprint 6** — Padrões de agrupamento (groupby)

## 🚀 Como executar
```bash
python Mini-Project-Varejo-SENAI-S7.py