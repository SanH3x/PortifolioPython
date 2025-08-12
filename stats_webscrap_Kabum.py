import pandas as pd
import matplotlib.pyplot as plt

# === 1. Leitura do CSV ===
arquivo = "precos_kabum.csv"  # Substitua pelo caminho do seu arquivo
df = pd.read_csv(arquivo)

# === 2. Limpeza e conversão da coluna de preços ===
def limpar_preco(valor):
    """
    Remove símbolos monetários, pontos e vírgulas e converte para float.
    Ex.: 'R$ 1.299,90' -> 1299.90
    """
    if pd.isna(valor):
        return None
    valor = str(valor)
    valor = valor.replace("R$", "").replace(".", "").replace(",", ".").strip()
    try:
        return float(valor)
    except ValueError:
        return None

df["Preço"] = df["Preço"].apply(limpar_preco)

# Remove linhas com preços inválidos
df = df.dropna(subset=["Preço"])

# === 3. Estatísticas básicas ===
estatisticas = {
    "Preço Médio": df["Preço"].mean(),
    "Mediana de Preços": df["Preço"].median(),
    "Preço Mínimo": df["Preço"].min(),
    "Preço Máximo": df["Preço"].max(),
    "Desvio Padrão": df["Preço"].std(),
    "Quantidade de Produtos": df["Produto"].count()
}

# Exibe estatísticas no console
for k, v in estatisticas.items():
    if k == "Quantidade de Produtos":
        print(f"{k}: {int(v)}")
    else:
        print(f"{k}: R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

# === 4. Produto mais barato e mais caro ===
produto_mais_barato = df.loc[df["Preço"].idxmin()]
produto_mais_caro = df.loc[df["Preço"].idxmax()]

print("\nProduto mais barato:")
print(f"{produto_mais_barato['Produto']} - R$ {produto_mais_barato['Preço']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

print("\nProduto mais caro:")
print(f"{produto_mais_caro['Produto']} - R$ {produto_mais_caro['Preço']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))


# === 5. Histograma de preços ===
plt.figure(figsize=(8,5))
plt.hist(df["Preço"], bins=15, color="#007acc", edgecolor="black")
plt.title("Distribuição de Preços - Kabum", fontsize=14)
plt.xlabel("Preço (R$)")
plt.ylabel("Frequência")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
