# Instale as dependências (caso não tenha)
# pip install nltk

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Baixar dados necessários do NLTK
nltk.download('vader_lexicon')

# Criar o analisador de sentimentos
sia = SentimentIntensityAnalyzer()

while True:
    texto = input("\nDigite um texto para análise (ou 'sair' para encerrar): ")
    if texto.lower() == 'sair':
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

    # Exibir resultado
    print("\nResultado da Análise:")
    print(f"Texto: {texto}")
    print(f"Sentimento: {sentimento}")
    print(f"Detalhes: {pontuacoes}")

