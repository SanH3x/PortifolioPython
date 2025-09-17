import pandas as pd

# Lista de produtos agrícolas
produtos = [
    {"Produto": "Herbicida Roundup Original da Monsanto", "Principio_Ativo": "Glifosato", "Quantidade": 20, "Unidade": "Litros"},
    {"Produto": "Bactericida Agrinose da Sumitomo", "Principio_Ativo": "Oxicloreto de Cobre", "Quantidade": 15, "Unidade": "Litros"},
    {"Produto": "Fungicida, Bactericida Cupronil da Funguran", "Principio_Ativo": "Oxicloreto de cobre", "Quantidade": 50, "Unidade": "Kg"},
    {"Produto": "Herbicida Arizona da Zhongshan", "Principio_Ativo": "Atrazina", "Quantidade": 30, "Unidade": "Litros"},
    {"Produto": "Inseticida da Landrin", "Principio_Ativo": "Indoxacarbe", "Quantidade": 100, "Unidade": "Unidades"},
    {"Produto": "Herbicida Preciso", "Principio_Ativo": "Glifosato", "Quantidade": 25, "Unidade": "Litros"},
]


# Criar DataFrame
df = pd.DataFrame(produtos)

# Exibir tabela organizada
print(df)


# Exemplo: filtrar apenas herbicidas
print("\nHerbicidas cadastrados:")
print(df[df["Produto"].str.contains("Fungicida")])
