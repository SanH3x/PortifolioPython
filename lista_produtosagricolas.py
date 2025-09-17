import pandas as pd

# Lista de produtos agrícolas
produtos = [
    {"Produto": "Herbicida A", "Principio_Ativo": "Glifosato", "Quantidade": 20, "Unidade": "Litros"},
    {"Produto": "Inseticida B", "Principio_Ativo": "Lambda-cialotrina", "Quantidade": 15, "Unidade": "Litros"},
    {"Produto": "Fungicida C", "Principio_Ativo": "Mancozebe", "Quantidade": 50, "Unidade": "Kg"},
    {"Produto": "Herbicida D", "Principio_Ativo": "Atrazina", "Quantidade": 30, "Unidade": "Litros"},
    {"Produto": "Inseticida E", "Principio_Ativo": "Imidacloprido", "Quantidade": 100, "Unidade": "Unidades"}
]

# Criar DataFrame
df = pd.DataFrame(produtos)

# Exibir tabela organizada
print(df)

# Exemplo: filtrar apenas herbicidas
print("\nHerbicidas cadastrados:")
print(df[df["Produto"].str.contains("Herbicida")])
