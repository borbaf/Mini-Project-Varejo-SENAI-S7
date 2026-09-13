# diagnostico_data.py — rodar antes de corrigir
import pandas as pd

# use os mesmos caminhos/parâmetros do Sprint 1
csv_path = r"C:\Users\borba\.cache\kagglehub\datasets\namespaiva\base-varejo\versions\1\Base Varejo.csv"
df = pd.read_csv(csv_path, encoding="latin1", sep=";")

print("Total:", len(df))
print("NaN reais      :", df["DATA"].isna().sum())

data_str = df["DATA"].astype(str).str.strip()
print("Vazias ('')    :", data_str.eq("").sum())
print("Marcadores     :", data_str.isin(["#N/D", "#N/A", "NA", "NULL", "null", "-"]).sum())

# Amostra dos valores que NÃO convertem no formato estrito
def converte_estrito(d: str) -> bool:
    return pd.notna(pd.to_datetime(d, format="%d/%m/%Y", errors="coerce"))

nao_convertem = data_str[~data_str.apply(converte_estrito)]
print(f"\nValores fora do formato dd/mm/aaaa: {len(nao_convertem)}")
print("Amostra:", nao_convertem.head(30).tolist())