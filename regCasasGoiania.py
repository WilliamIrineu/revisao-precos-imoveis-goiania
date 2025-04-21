

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv('casagoiania.csv')

df.head()

df.shape

df.info()

df.isna().sum()

# Tratamento de dados

def convert_money(txt):
  money = txt.strip('R$ ').replace('.', '').replace(',', '.')
  return float(money)

df['PRICE'].unique()

df = df[df['PRICE'] != 'Sob consulta']

df['PRICE'] = df['PRICE'].apply(lambda x : convert_money(x))

df['PRICE'].describe()

df['PRICE'].unique()
df.info()

df['CONDOMÍNIO'].unique()

df['CONDOMÍNIO'] = df['CONDOMÍNIO'].apply(lambda x : convert_money(x) 
                                          if pd.isnull(x) != True else float(0))

df.isna().sum()

df['IPTU'] = df['IPTU'].apply(lambda x : convert_money(x) 
                              if pd.isnull(x) != True else np.nan)
df.isna().sum()


df['BEDROOMS'].unique()
df['PARKING-SPACES'].unique()
df['BATHROOMS'].unique()

for i in ['BEDROOMS','PARKING-SPACES', 'BATHROOMS']:
    df[i] = df[i] = pd.to_numeric(df[i], errors='coerce').astype('Int64')

df.dropna(subset=['ADDRESS', 'AREAS'], inplace=True)

df.isna().sum()

df['AREAS'] = df.AREAS.str.replace(' m²', '').str.split(' - ').apply(lambda x: [int(i) for i in x])
df['AREAS'] = df['AREAS'].apply(np.mean)

df.isna().sum()

#1. Preenchimento de valores ausentes em IPTU com interpolação:
df.IPTU.interpolate(limit_direction='both', inplace=True)

# 2. Preenchimento de BEDROOMS baseado no tipo de imóvel:
m = (df['BEDROOMS'].isna()) & (df['TIPO'] == 'fazendas-sitios-chacaras')
df.loc[m,'BEDROOMS'] = df.loc[m,'BEDROOMS'].fillna(1)

m1 = (df['BEDROOMS'].isna()) & (df['TIPO'] == 'apartamentos')
df.loc[m1,'BEDROOMS'] = df.loc[m1,'BEDROOMS'].fillna(3)

m2 = (df['BEDROOMS'].isna()) & (df['TIPO'] == 'casas')
df.loc[m2,'BEDROOMS'] = df.loc[m2,'BEDROOMS'].fillna(3)

m3 = (df['BEDROOMS'].isna()) & (df['TIPO'] == 'quitinetes')
df.loc[m3,'BEDROOMS'] = df.loc[m3,'BEDROOMS'].fillna(1)

m4 = (df['BEDROOMS'].isna()) & (df['TIPO'] == 'terrenos-lotes-condominios')
df.loc[m4,'BEDROOMS'] = df.loc[m4,'BEDROOMS'].fillna(1)


# Pré-processamento para uso do KNNImputer:
df.BATHROOMS = df.BATHROOMS.replace({np.nan: np.nan})
df['PARKING-SPACES'] = df['PARKING-SPACES'].replace({np.nan: np.nan})

#Imputação de valores faltantes com KNNImputer

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
for c in ['BATHROOMS', 'PARKING-SPACES']:
    x = imputer.fit_transform(df[c].values.reshape(-1, 1))
    df[c] = x

#5. Arredonda os valores de BATHROOMS e PARKING-SPACES:
df['BATHROOMS'] = np.round(df['BATHROOMS'], 0)
df['PARKING-SPACES'] = np.round(df['PARKING-SPACES'], 0)

# 6. Conversão de BEDROOMS para float:
df['BEDROOMS'] = df['BEDROOMS'].astype(float)

# 7. Exibe resumo dos valores ausentes:
print('Resume Missing Values')
df.isnull().sum().sort_values(ascending=False)

df.shape
df.columns
df.info()
df.describe()
print(df['TIPO'].unique())

print(df['ADDRESS'].unique())
df['ADDRESS']

# Expressão regular para capturar nomes de setores ou bairros
regex_setor = r'(Setor\s+\w+|Bairro\s+\w+|Jardim\s+\w+|Parque\s+\w+|Chácaras\s+\w+)'

# Criação da nova coluna
df['SECTOR'] = df['ADDRESS'].str.extract(regex_setor, expand=False)
df['SECTOR']

df_original = df.copy()

df.drop(['DATE', 'ADDRESS'], axis=1,inplace=True)

df.shape
df.columns

# ANALISE DESCRITIVA
colors = sns.cubehelix_palette(reverse=True)
colors

# CONTAGEM POR TIPO
fig, ax1 = plt.subplots(1, 1, figsize=(10, 3), dpi=102)
df.TIPO.value_counts().plot(kind='bar', ax=ax1, color=colors[0])
ax1.set_title('TIPO: counts', fontsize=10)
ax1.axhline(linewidth=4, color="black")
ax1.tick_params(labelsize=8)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8, 
         rotation_mode="anchor")


# CORRELACAO

cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=60)
sns.heatmap(df.drop(columns=['TIPO', 'SECTOR']).corr(), annot=True, cmap=cmap, ax=ax1)
ax1.set_title('Correlations', fontsize=16)
ax2.set_title('Distributions log(PRICE)', fontsize=16)
sns.histplot(np.log1p(df.PRICE), kde=True, color='blue', ax=ax2) 


# CONTAGEM POR SECTOR
# Conta os setores e pega os 10 mais frequentes
top10_sectors = df.SECTOR.value_counts().head(10)

# Gráfico
fig, ax1 = plt.subplots(1, 1, figsize=(10, 4), dpi=102)
top10_sectors.plot(kind='bar', ax=ax1, color='royalblue')  # ou colors[0] se estiver definido

ax1.set_title('Top 10 SECTORs mais frequentes', fontsize=12)
ax1.set_ylabel('Contagem')
ax1.axhline(linewidth=2, color="black", alpha=0.3)
ax1.tick_params(labelsize=9)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=9, rotation_mode="anchor")
plt.tight_layout()
plt.show()




df.hist(bins = 30, figsize=(20,20), color = 'r')

df.columns
df.dtypes
df.head()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['PRICE'])
plt.title('Boxplot dos Preços dos Imóveis')
plt.xlabel('Preço (R$)')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(np.log1p(df.PRICE), bins=50, kde=True, color='orange')
plt.title('Distribuição dos Preços (Log Transformado)')
plt.xlabel('log(1 + Preço)')
plt.ylabel('Frequência')
plt.show()

Q1 = df['PRICE'].quantile(0.25)
Q3 = df['PRICE'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtra os dados sem outliers
df= df[(df['PRICE'] >= limite_inferior) & (df['PRICE'] <= limite_superior)]

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['PRICE'])
plt.title('Boxplot dos Preços dos Imóveis')
plt.xlabel('Preço (R$)')
plt.show()


Q1 = df['PRICE'].quantile(0.25)
Q3 = df['PRICE'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtra os dados sem outliers
df= df[(df['PRICE'] >= limite_inferior) & (df['PRICE'] <= limite_superior)]

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['PRICE'])
plt.title('Boxplot dos Preços dos Imóveis')
plt.xlabel('Preço (R$)')
plt.show()


Q1 = df['PRICE'].quantile(0.25)
Q3 = df['PRICE'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtra os dados sem outliers
df= df[(df['PRICE'] >= limite_inferior) & (df['PRICE'] <= limite_superior)]

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['PRICE'])
plt.title('Boxplot dos Preços dos Imóveis')
plt.xlabel('Preço (R$)')
plt.show()


Q1 = df['PRICE'].quantile(0.25)
Q3 = df['PRICE'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtra os dados sem outliers
df= df[(df['PRICE'] >= limite_inferior) & (df['PRICE'] <= limite_superior)]

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['PRICE'])
plt.title('Boxplot dos Preços dos Imóveis')
plt.xlabel('Preço (R$)')
plt.show()

Q1 = df['PRICE'].quantile(0.25)
Q3 = df['PRICE'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtra os dados sem outliers
df= df[(df['PRICE'] >= limite_inferior) & (df['PRICE'] <= limite_superior)]

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['PRICE'])
plt.title('Boxplot dos Preços dos Imóveis')
plt.xlabel('Preço (R$)')
plt.show()


df.shape

plt.figure(figsize=(10, 8))
sns.heatmap(df[['PRICE', 'BEDROOMS', 'BATHROOMS', 'PARKING-SPACES', 'AREAS', 'CONDOMÍNIO', 'IPTU']].corr(), 
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()


# Pre processamento

df.columns
df.head()

df.shape

df.isna().sum()
df.dropna(inplace=True)

df.isna().sum()
df.shape


X_cat=df[['TIPO', 'SECTOR']]

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(sparse_output=False)

X_cat_encoded = onehotencoder.fit_transform(X_cat)
col_names = onehotencoder.get_feature_names_out(X_cat.columns)
X_cat = pd.DataFrame(X_cat_encoded, columns=col_names)
X_cat

X_numerical = df[['BEDROOMS', 'PARKING-SPACES', 'BATHROOMS','AREAS','CONDOMÍNIO', 'IPTU']]
X_numerical

X_cat = X_cat.reset_index(drop=True)
X_numerical = X_numerical.reset_index(drop=True)


X_all = pd.concat([X_cat, X_numerical], axis = 1)
X_all.columns = X_all.columns.astype(str) # ATUALIZAÇÃO JAN-205
X_all


X=X_all
X
y = np.log1p(df.PRICE)
y

#variaveis_significantes = ['TIPO_apartamentos', 'TIPO_casas', 'TIPO_casas-de-condominio',
#       'TIPO_cobertura', 'TIPO_flat', 'TIPO_terrenos-lotes-condominios',
#       'SECTOR_Jardim América', 'SECTOR_Jardim Atlântico',
#       'SECTOR_Jardim Europa', 'SECTOR_Jardim Goiás', 'SECTOR_Jardim Novo',
#       'SECTOR_Parque Amazônia', 'SECTOR_Parque Oeste',
#       'SECTOR_Setor Aeroporto', 'SECTOR_Setor Bela', 'SECTOR_Setor Bueno',
#       'SECTOR_Setor Central', 'SECTOR_Setor Faiçalville',
#       'SECTOR_Setor Leste', 'SECTOR_Setor Marista', 'SECTOR_Setor Negrão',
#       'SECTOR_Setor Oeste', 'SECTOR_Setor Pedro', 'SECTOR_Setor Sudoeste',
#       'SECTOR_Setor Sul', 'BEDROOMS', 'PARKING-SPACES', 'BATHROOMS', 'AREAS',
#       'CONDOMÍNIO', 'IPTU']

# Remove essas colunas de X
#X= X[variaveis_significantes]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=500,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)


rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
mae = mean_absolute_error(y_test_real, y_pred)
mape = mean_absolute_percentage_error(y_test_real, y_pred) * 100
r2 = r2_score(y_test_real, y_pred)

print(f"RMSE: R$ {rmse:.2f}")
print(f"MAE: R$ {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")


#RMSE: R$ 90613.03
#MAE: R$ 46967.16
#MAPE: 17.60%
#R²: 0.8641


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cria uma série com os nomes e importâncias
feat_importance = pd.Series(model.feature_importances_, index=X.columns)

# Seleciona as 22 mais importantes
top_feats = feat_importance.sort_values(ascending=False).head(20)
top_feats


# Gráfico
plt.figure(figsize=(10, 6))
sns.barplot(x=top_feats.values, y=top_feats.index, orient='h')
plt.title("Top 20 Variáveis Mais Importantes - Random Forest")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.tight_layout()
plt.show()


# técnica de seleção de variáveis baseada na variância,

from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)  # ou 0.05 dependendo do caso
X_filtrado = selector.fit_transform(X)
colunas_selecionadas = X.columns[selector.get_support()]
print(colunas_selecionadas)


import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns

# Extrai importâncias de cada árvore
all_importances = np.array([tree.feature_importances_ for tree in model.estimators_])  # shape: (n_arvores, n_features)

# Média e desvio padrão por variável
means = all_importances.mean(axis=0)
stds = all_importances.std(axis=0)
n = all_importances.shape[0]
conf_int = t.ppf(0.975, df=n-1) * stds / np.sqrt(n)

# DataFrame com ICs
df_ic = pd.DataFrame({
    'variavel': X.columns,
    'importancia_media': means,
    'ic_inf': means - conf_int,
    'ic_sup': means + conf_int,
})

# Marcar se inclui o zero no intervalo de confiança
df_ic['inclui_zero'] = (df_ic['ic_inf'] <= 0) & (df_ic['ic_sup'] >= 0)

# Filtrar as 40 mais importantes
df_top = df_ic.sort_values('importancia_media', ascending=False).head(32)

# Plot
plt.figure(figsize=(10, 8))
plt.errorbar(df_top['importancia_media'], df_top['variavel'], 
             xerr=conf_int[df_top.index], fmt='o', color='blue', ecolor='gray', capsize=3)
plt.axvline(0, color='red', linestyle='--', label='Zero')
plt.title("Intervalo de Confiança (95%) das Importâncias - Top 32 Variáveis")
plt.xlabel("Importância média")
plt.ylabel("Variável")
plt.legend()
plt.tight_layout()
plt.show()

# Teste de hipótese: importância ≠ 0
df_ic['t_stat'] = df_ic['importancia_media'] / (stds / np.sqrt(n))
df_ic['p_valor'] = 2 * (1 - t.cdf(np.abs(df_ic['t_stat']), df=n-1))
df_ic['significativo_5%'] = df_ic['p_valor'] < 0.05

# Resultado: todas variáveis com p < 0.05
df_significativas = df_ic[df_ic['significativo_5%']]

# Exibir ou salvar
print(df_significativas[['variavel', 'importancia_media', 'p_valor']].sort_values('importancia_media', ascending=False))



# validacao cruzadas random florest

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

model = RandomForestRegressor(
    n_estimators=500,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# Previsões cruzadas no log
y_pred_log = cross_val_predict(model, X_train, y_train, cv=10, n_jobs=-1)

# Reverter o log1p
y_real = np.expm1(y_train)
y_pred = np.expm1(y_pred_log)

# Métricas em reais
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
mae = mean_absolute_error(y_real, y_pred)
mape = mean_absolute_percentage_error(y_real, y_pred) * 100
r2 = r2_score(y_real, y_pred)

print(f"✅ Erro médio REAL em R$:")
print(f"RMSE: R$ {rmse:,.2f}")
print(f"MAE : R$ {mae:,.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²  : {r2:.4f}")



model = RandomForestRegressor(
    n_estimators=500,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

import matplotlib.pyplot as plt
import seaborn as sns

# Dispersão
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test_real, y=y_pred, alpha=0.3)

# Linha de perfeição
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min(), y_test_real.max()], 'r--', label='Perfeito (y = x)')

# Faixa de confiança de ±20%
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min()*0.8, y_test_real.max()*0.8], 'g--', alpha=0.7, label='-20%')
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min()*1.2, y_test_real.max()*1.2], 'g--', alpha=0.7, label='+20%')

plt.xlabel('Preço Real (R$)')
plt.ylabel('Preço Previsto (R$)')
plt.title('Comparação: Preços Reais vs. Previstos com Intervalo de Confiança (±20%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




import joblib

# Salvar o modelo
joblib.dump(model, 'modelo_random_forest.pkl')

# Recarregar o modelo
modelo_carregado = joblib.load('modelo_random_forest.pkl')


# novo imóvel com os dados:

novo_imovel = X.iloc[0]
novo_imovel

pd.DataFrame([novo_imovel], columns=X.columns)

# Fazer a previsão (convertendo para DataFrame com uma linha)
preco_log = modelo_carregado.predict(pd.DataFrame([novo_imovel], columns=X.columns))

# Desfaz o log (caso você tenha usado log1p/y_log = log(1 + y))
preco_estimado = np.expm1(preco_log)

# Mostra o resultado
print(f"Preço estimado: R$ {preco_estimado[0]:,.2f}")
print(f"Resultado real: R$ {np.expm1(y.iloc[0]):,.2f}")



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf_model = RandomForestRegressor(random_state=42)

rf_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 1.0]
}

rf_random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_param_grid,
    n_iter=30,
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

rf_random_search.fit(X_train, y_train)

print("Melhores parâmetros RandomForest:")
print(rf_random_search.best_params_)
print("Score:")
print(rf_random_search.best_score_)

