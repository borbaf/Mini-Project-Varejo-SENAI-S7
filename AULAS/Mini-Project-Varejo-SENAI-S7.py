import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "Base Varejo.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "namespaiva/base-varejo",
  file_path,
  pandas_kwargs={
      "encoding": "latin1",
      "sep": None,            # <-- detecta o separador automaticamente
      "engine": "python",     # <-- necessário para o sep=None
      "on_bad_lines": "skip", # <-- ignora linhas problemáticas, se houver
  },
)

print("Shape:", df.shape)
print("Colunas:", df.columns.tolist())
print(df.head())