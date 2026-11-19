
# 1. Importar as bibliotecas
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

# 2. Ler a base de dados
df = pd.read_csv("exemplo_regressao_CPU_tempo_resposta.csv")

# Separar as variáveis
X = df["CPU_medio_pct"]
y = df["Tempo_resposta_ms"]


# 3. Associação - Correlação de Pearson
r, pvalor = pearsonr(X, y)

print("CORRELAÇÃO DE PEARSON")
print(f"Correlação (r): {r:.4f}") 
print(f"p-valor: {pvalor:.6f}") #se for menor que 0,05 não é um acontecimento por acaso 


# 4. Ajuste do modelo de regressão
X_design = sm.add_constant(X)
modelo = sm.OLS(y, X_design).fit()

print("\nRESUMO DO MODELO")
print(modelo.summary())


# 5. Interpretação de B0 e B1
b0 = modelo.params["const"]
b1 = modelo.params["CPU_medio_pct"]

print("\nINTERPRETAÇÃO DOS COEFICIENTES")
print(f"B0 (intercepto): {b0:.4f}")
print(f"B1 (inclinação): {b1:.4f}")

print(f"Tempo estimado quando CPU = 0%: {b0:.4f} ms")
print(f"Para cada aumento de 1 ponto percentual na CPU,")
print(f"o tempo estimado varia {b1:.4f} ms.")


# 6. Valores estimados e resíduos
df["y_estimado"] = modelo.fittedvalues
df["residuo"] = modelo.resid

print("\nVALORES ESTIMADOS E RESÍDUOS")
print(df[[
    "CPU_medio_pct",
    "Tempo_resposta_ms",
    "y_estimado",
    "residuo"
]].head(15).round(3))
