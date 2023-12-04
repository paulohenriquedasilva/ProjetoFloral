from flask import Flask, jsonify, render_template, request, redirect
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# import os

# dados = pd.read_csv("IRIS.csv")

# #Transformação de Dados
# variavel_explicativa = dados.drop('species', axis = 1)
# variavel_alvo = dados['species']



# #Transformando a variavel alvo
# label_encoder = LabelEncoder()
# variavel_alvo = label_encoder.fit_transform(variavel_alvo)

# #Divisão de treino e teste
# variavel_explicativa_treino, variavel_explicativa_teste, variavel_alvo_treino, variavel_alvo_teste = train_test_split(variavel_explicativa, variavel_alvo, stratify= variavel_alvo)


# #Árvore de Decisão
# arvore = DecisionTreeClassifier()
# arvore.fit(variavel_explicativa_treino, variavel_alvo_treino)
# arvore.predict(variavel_explicativa_teste)
# arvore.score(variavel_explicativa_teste, variavel_alvo_teste)


# #plt.figure(figsize = (15, 6))
# plot_tree(arvore, filled = True);

# ##ajustando modelo

# arvore = DecisionTreeClassifier(max_depth = 3)
# arvore.fit(variavel_explicativa_treino, variavel_alvo_treino)
# print(arvore.score(variavel_explicativa_treino, variavel_alvo_treino))
# #plt.figure(figsize = (15, 6))
# plot_tree(arvore, filled = True);

# print(f'Acurácia Árvore: {arvore.score(variavel_explicativa_teste, variavel_alvo_teste)}')

# with open('modelo_arvore1.pkl', 'wb') as arquivo_machineLearning:
#   pickle.dump(arvore, arquivo_machineLearning)


decision_tree_model = joblib.load(r'C:\\Users\Rodrigo\Desktop\IA_FLORAL\modelo_arvore1.pkl')

# species_mapping = {
#     0: 'Iris Setosa',
#     1: 'Iris Versicolor',
#     2: 'Iris Virginica'
# }


app = Flask(__name__)

app.static_folder = 'static'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Iris')
def iris():
    return render_template('about-iris.html')


@app.route('/About')
def about():
    return render_template('about-us.html')

@app.route('/predict', methods=['POST', ])
def predict():
    if request.method == 'POST':
        sepalLength = float(request.form['sepalLength'])
        sepalWidth = float(request.form['sepalWidth'])
        petalLength = float(request.form['petalLength'])
        petalWidth = float(request.form['petalWidth'])

        
        input_data = np.array([[sepalLength, sepalWidth, petalLength, petalWidth]]) 

        #Previsão
        prediction = decision_tree_model.predict(input_data)

        #Mapeando a saída numérica para o nome da espécie correspondente

        if prediction == 0:
            result = 'Iris Setosa'
        elif prediction == 1:
            result = 'Iris Versicolor'
        elif prediction == 2:
            result = 'Iris Virginica'
        else:
            result = 'Espécie não catalogada pelo sistema'

        #speceis_result = species_mapping.get(prediction[0], 'Espécie Desconhecida')

        return render_template('index.html', result=result)#speceis_result)



if __name__ == '__main__':
    app.run(debug=True)
