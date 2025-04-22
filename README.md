# Previsão de Preços de Imóveis em Goiânia-GO com Machine Learning

Este projeto utiliza técnicas de Machine Learning para estimar o preço de venda de imóveis na cidade de Goiânia-GO, com base em dados reais extraídos via web scraping do site ZAP Imóveis.

## 🔍 Visão Geral

- Coleta e tratamento de dados brutos
- Análise exploratória e visualizações
- Limpeza e preenchimento de dados ausentes
- Criação de variáveis categóricas com OneHotEncoding
- Modelagem com Random Forest Regressor
- Avaliação por hold-out e validação cruzada
- Exportação do modelo treinado para aplicação web

## 📁 Dataset

Os dados foram extraídos do portal ZAP Imóveis em agosto de 2021. O conjunto de dados inclui:

- Preço do imóvel (R\$)
- Endereço
- Área (m²)
- Número de quartos
- Banheiros
- Vagas de garagem
- Valor do Condomínio
- Valor do IPTU
- Tipo do imóvel

## 📈 Resultados

O modelo final apresentou os seguintes resultados:

| Métrica                          | Teste Final (Hold-Out) | Validação Cruzada (10-Fold) |
|----------------------------------|-------------------------|------------------------------|
| **RMSE (Erro Quadrático Médio)** | R$ 75.752,20            | R$ 84.784,67                 |
| **MAE (Erro Absoluto Médio)**    | R$ 40.832,06            | R$ 44.961,02                 |
| **MAPE (Erro Percentual Médio)** | 8,91%                   | 9,81%                        |
| **R² (Coeficiente de Determinação)** | 0,8975              | 0,8825                       |


## 🌐 Aplicativo Web com Flask

Uma versão web simples foi criada usando o framework Flask. Com isso, é possível simular o preço de um imóvel diretamente no navegador.


## 🎓 Tecnologias

- Python 3.10+
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- joblib
- Flask (para aplicação web)

## 🚀 Como usar o modelo treinado

```python
import joblib
modelo = joblib.load('modelo_random_forest.pkl')
preco_log = modelo.predict(novo_dado)
preco = np.expm1(preco_log)
```

Este projeto foi desenvolvido com fins educacionais, analíticos e práticos, podendo ser expandido para outros contextos e cidades.

Contribuições e melhorias são bem-vindas!

📑 Autor: WILLIAM IRINEU

