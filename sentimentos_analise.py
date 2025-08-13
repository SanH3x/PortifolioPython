
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Baixar dados necessários do NLTK
nltk.download('vader_lexicon')

# Criar o analisador de sentimentos
sia = SentimentIntensityAnalyzer()

# Nome do arquivo de relatório com data/hora para evitar sobrescrita
arquivo_relatorio = f"relatorio_sentimento_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

# Função para salvar no relatório
def salvar_relatorio(texto, sentimento, pontuacoes):
    with open(arquivo_relatorio, "a", encoding="utf-8") as f:
        f.write(f"Texto: {texto}\n")
        f.write(f"Sentimento: {sentimento}\n")
        f.write(f"Pontuações: {pontuacoes}\n")
        f.write("-" * 50 + "\n")

print("=== Analisador de Sentimento ===")
print("Digite 'sair' para encerrar.\n")

while True:
    texto = input("Digite um texto: ")
    if texto.lower() == 'sair':
        print(f"\nRelatório salvo em: {arquivo_relatorio}")
        break

    # Obter pontuações
    pontuacoes = sia.polarity_scores(texto)

    # Determinar sentimento
    if pontuacoes['compound'] >= 0.05:
        sentimento = "Positivo"
    elif pontuacoes['compound'] <= -0.05:
        sentimento = "Negativo"
    else:
        sentimento = "Neutro"

    # Exibir resultado no console
    print(f"Sentimento: {sentimento} | Pontuações: {pontuacoes}")

    # Salvar no relatório
    salvar_relatorio(texto, sentimento, pontuacoes)
