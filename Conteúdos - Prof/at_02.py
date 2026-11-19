from statsmodels.stats.proportion import proportions_ztest

aprovados = [34, 53]    # A, B
tamanhos  = [140, 230]

z, p = proportions_ztest(aprovados, tamanhos)  # bilateral por padrão

print(f"z = {z:.4f}")
print(f"valor p (bilateral) = {p:.4f}")

alfa = 0.10
print("Rejeita H0" if p < alfa else "Não rejeita H0")