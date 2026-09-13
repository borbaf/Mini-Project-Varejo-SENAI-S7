import kagglehub
import zipfile
import pandas as pd
import os

path = kagglehub.dataset_download("namespaiva/base-varejo")

# Encontra o arquivo que é um ZIP (mesmo com nome .csv)
zip_path = None
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith(".csv"):  # o "Base Varejo.csv" é na verdade um zip
            zip_path = os.path.join(root, f)
            break

print("Arquivo encontrado:", zip_path)

# Extrai o conteúdo do ZIP para uma pasta "extraido"
destino = os.path.join(os.path.dirname(zip_path), "extraido")
os.makedirs(destino, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(destino)
    print("Conteúdo do ZIP:", z.namelist())

# Lê o CSV real de dentro do ZIP
csv_real = os.path.join(destino, "Base Varejo.csv")  # ajuste o nome conforme o namelist()
df = pd.read_csv(csv_real, encoding="latin1", sep=";")

print("\nShape:", df.shape)
print("Colunas:", df.columns.tolist())
print(df.head())