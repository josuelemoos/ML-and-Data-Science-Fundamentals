
# %%

import pandas as pd

df = pd.read_excel("data/dados_cerveja.xlsx") #importando os dados para um dataframe
df.head()   #mostrando as primeiras linhas do dataframe

# %%

features = ['temperatura','copo','espuma','cor'] #caracteristicas
target = 'classe' #alvo
#divindo o que queremos encontrar (target) do que vamos usar para encontrar (features)


X = df[features]
y = df[target]

X = X.replace({
    "mud": 1, "pint":2,
    "sim":1, "não":0,
    "clara": 0, "escura":1,
}) #trocando valores de texto para numeros que podem ser previstos

X
# %%

from sklearn import tree

model = tree.DecisionTreeClassifier() #criando o modelo de arvore de decisao
model.fit(X=X, y=y) # treinando o modelo com os dados de treino

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features,
               class_names=model.classes_,
               filled=True
               ) #mostrando arvore de decisao
# %%
