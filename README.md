# Mini-Project-Varejo-SENAI-S7

Análise Exploratória de Dados (AED) da **Base Varejo**, desenvolvida como
Mini-Projeto Avaliativo (Módulo 1, Semana 07) — Turma Analise_de_Dados_T6.

## 📌 Autor
- **Filippe Borba**

## 🗂️ Base de dados
- **Dataset**: [Base Varejo](https://www.kaggle.com/datasets/namespaiva/base-varejo)
- **Origem**: Kaggle | 830.000 registros | 14 colunas | período 2019-01 a 2022-12
- **Download automático** via `kagglehub` (sem baixar manualmente)

## 📖 Dicionário de dados
Fonte: referência da base (Projeto III — FATEC Rubens Lara, seção I.1.2).

| Campo | Descrição |
|-------|-----------|
| DATA | Data da compra |
| CO_ID | Identificação do número de compra (nota fiscal) — repetido por item |
| CL_ID | Identificação do cliente |
| CL_GENERO | Sexo biológico informado pelo cliente |
| CL_EC | Estado civil: 1 Casado/união estável; 2 Divorciado; 3 Separado; 4 Solteiro; 5 Viúvo |
| CL_FHL | Número de filhos do cliente |
| CL_SEG | Segmentação econômica (classe A, B ou C) |
| PR_ID | Código do produto (SKU) |
| PR_CAT | Categoria do produto (vazio/`#N/D` → "SEM CATEGORIA") |
| PR_NOME | Nome do produto |

## 🧠 Reflexão teórica — ETL e qualidade de dados

**Extract**: a base foi obtida automaticamente via `kagglehub`, sem download manual,
e carregada com `pandas.read_csv`, preservando tipos e estrutura original.

**Transform**: aplicamos limpeza de texto (remoção de espaços e normalização),
conversão de tipos (inteiros via função tolerante, `DATA` para `datetime` com
`dayfirst=True` — interpretação correta do padrão brasileiro) e tratamento de
categoria ausente (`#N/D` → "SEM CATEGORIA").

**Qualidade de dados**: identificamos e tratamos valores nulos (colunas vazias
100% removidas), eliminamos duplicatas relevantes (96.553 linhas → 11,6% da base,
justificando a escolha) e validamos a regra de negócio do identificador de compra:
cada linha representa UM item comprado, com `CO_ID` repetido entre os itens de uma
mesma compra — por isso agrupamentos usam `groupby` sem perder o grão de item.

**Load**: o DataFrame consolidado (733.447 linhas × 10 colunas, sem nulos) fica
pronto para estatísticas, agrupamentos e futuras análises/dashboards.

## 💡 Insights (3–6 tópicos)

1. **Qualidade dos dados**: 830.000 → 733.447 registros após remoção de 96.553
   duplicatas (11,6%); colunas 100% vazias eliminadas; datas convertidas sem nulos.
2. **Perfil do cliente — filhos (CL_FHL)**: média 1,15, mediana 0, moda 0,
   desvio 1,42, faixa 0–4 — distribuição assimétrica à direita: maioria sem filhos.
3. **Gênero**: público feminino domina — 52,1% dos itens (F) vs 47,9% (M).
4. **Categorias**: ALIMENTOS lidera com 52,4% dos itens, seguida de HIGIENE (18,8%)
   e LIMPEZA (17,5%).
5. **Sazonalidade**: pico de 28.575 itens em 2021-10; média mensal ~15.280 itens.
6. **Problema remanescente**: queda abrupta em 2022-09/10 (1.297 e 2.373 itens) —
   provável coleta incompleta no fim da série, não sazonalidade real; e a base não
   possui coluna de valor, limitando análises a volume de itens.

## 🚀 Como executar

Pré-requisito: Python 3.10+ com `venv`. Instale as dependências com
`pip install -r requirements.txt` e execute `python Mini-Project-Varejo-SENAI-S7.py`.
No Google Colab, rode o mesmo arquivo em uma célula após `!pip install -r requirements.txt`.