# diagnostico_data.py — rodar antes de corrigir
import pandas as pd

csv_path = r"C:\Users\borba\.cache\kagglehub\datasets\namespaiva\base-varejo\versions\1\Base Varejo.csv"
df = pd.read_csv(csv_path, encoding="latin1", sep=";")

# Teste 1: conversão tolerante (dayfirst=True)
conv = pd.to_datetime(df["DATA"], dayfirst=True, errors="coerce")
print("NaT com dayfirst=True:", conv.isna().sum())

# Teste 2: padrões dos que ainda falharem
mascara = conv.isna()
print("Padrões (10 primeiros caracteres):")
print(df.loc[mascara, "DATA"].astype(str).str[:10].value_counts().head(20))