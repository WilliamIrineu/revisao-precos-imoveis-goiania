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

| Métrica | Teste Hold-Out | Validação Cruzada |
| ------- | -------------- | ----------------- |
| RMSE    | R\$ 90.613,03  | R\$ 94.458,08     |
| MAE     | R\$ 46.967,16  | R\$ 49.433,57     |
| MAPE    | 17,60%         | 14,33%            |
| R²      | 0.8641         | 0.8562            |

## 🌐 Aplicativo Web com Flask

Uma versão web simples foi criada usando o framework Flask. Com isso, é possível simular o preço de um imóvel diretamente no navegador.

### Como rodar a aplicação:

```bash
# Instale as dependências
pip install -r requirements.txt

# Rode o servidor local
python app.py

# Acesse em:
http://127.0.0.1:5000
```

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

## 🔗 Links

- Notebook no Google Colab
- [Código completo no GitHub](https://github.com/seuusuario/previsao-preco-imoveis-goiania)

---

Este projeto foi desenvolvido com fins educacionais, analíticos e práticos, podendo ser expandido para outros contextos e cidades.

Contribuições e melhorias são bem-vindas!

📑 Autor: Lucas Gabriel

