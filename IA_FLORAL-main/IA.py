# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# import joblib

# # Carregue a base de dados
# df = pd.read_csv('IRIS.csv')

# # Separe os dados em recursos (X) e rótulos (y)
# X = df.iloc[:, :-1].values  # Assume que a última coluna é o rótulo
# y = df.iloc[:, -1].values

# # Separe os dados em conjuntos de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalize os dados
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Crie e treine o modelo KNN
# knn_model = KNeighborsClassifier(n_neighbors=3)
# knn_model.fit(X_train_scaled, y_train)

# # Salve o modelo e o scaler para uso posterior
# joblib.dump(knn_model, 'Rodrigo\\Desktop\\IA_FLORAL\\modelo_treinado.pkl')

# joblib.dump(scaler, 'Rodrigo\\Desktop\\IA_FLORAL\\scaler.pkl')

