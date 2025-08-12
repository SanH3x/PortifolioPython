from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

# Configuração do Selenium (modo headless para não abrir janela)
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Inicializar o driver
driver = webdriver.Chrome(options=chrome_options)

# URL de pesquisa (exemplo: placa de vídeo)
url = "https://www.kabum.com.br/hardware/placa-de-video-vga"
driver.get(url)

# Espera para carregamento dinâmico
time.sleep(5)

# Lista para armazenar dados
produtos = []

# Localizar os elementos dos produtos
items = driver.find_elements(By.CSS_SELECTOR, "article.productCard")

for item in items:
    try:
        nome = item.find_element(By.CSS_SELECTOR, "span.nameCard").text
    except:
        nome = "Não encontrado"
    
    try:
        preco = item.find_element(By.CSS_SELECTOR, "span.priceCard").text
    except:
        preco = "Não informado"
    
    produtos.append({"Produto": nome, "Preço": preco})

# Fechar navegador
driver.quit()

# Criar DataFrame
df = pd.DataFrame(produtos)

# Exibir
print(df)

# Salvar em CSV
df.to_csv("precos_kabum.csv", index=False, encoding="utf-8-sig")
