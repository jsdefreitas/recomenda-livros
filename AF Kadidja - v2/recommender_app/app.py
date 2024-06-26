# Importações necessárias para o aplicativo Flask e manipulação de dados
from flask import Flask, render_template, jsonify
import pandas as pd
import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Definição dos caminhos dos arquivos CSV contendo os dados
base_dir = os.path.dirname(os.path.abspath(__file__))  # Diretório base do arquivo atual
user_reading_history_path = os.path.join(base_dir, '..', 'user_reading_history.csv')  # Caminho para o histórico de leitura dos usuários
book_metadata_path = os.path.join(base_dir, '..', 'book_metadata.csv')  # Caminho para os metadados dos livros

# Verificação da existência dos arquivos necessários
if not os.path.exists(user_reading_history_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {user_reading_history_path}")

if not os.path.exists(book_metadata_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {book_metadata_path}")

# Carregamento dos dados de leitura do arquivo CSV para um DataFrame pandas
df = pd.read_csv(user_reading_history_path)

# Criação de uma matriz esparsa de usuários e livros a partir do DataFrame
user_book_matrix = df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

# Conversão da matriz para o formato csr_matrix para eficiência computacional
user_book_matrix_sparse = csr_matrix(user_book_matrix.values)

# Configuração do modelo kNN (K-Nearest Neighbors) para recomendações colaborativas
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_book_matrix_sparse)

# Carregamento dos metadados dos livros do arquivo CSV
books_df = pd.read_csv(book_metadata_path)

# Tratamento de valores NaN na coluna 'description' dos metadados dos livros
books_df['description'].fillna('', inplace=True)

# Treinamento do modelo TF-IDF para análise de similaridade entre descrições de livros
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['description'].astype(str))  # Conversão explícita para string

# Cálculo da similaridade de cosseno entre os vetores TF-IDF
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Inicialização do aplicativo Flask
app = Flask(__name__)

# Rota principal que renderiza o template index.html
@app.route('/')
def index():
    return render_template('index.html')

# Rota para obter recomendações colaborativas para um usuário específico
@app.route('/recommendations/<int:user_id>')
def get_collaborative_recommendations(user_id):
    if user_id not in user_book_matrix.index:
        return jsonify([])  # Retorna uma lista vazia se o usuário não existir
    
    n_users = user_book_matrix.shape[0]
    n_recommendations = min(n_users - 1, 5)  # Limita o número de recomendações ao máximo possível
    
    user_index = user_book_matrix.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(user_book_matrix_sparse[user_index], n_neighbors=n_recommendations+1)
    
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        index = indices.flatten()[i]
        if index < len(user_book_matrix.columns):
            book_id = user_book_matrix.columns[index]
            book_title = books_df.loc[books_df['book_id'] == book_id, 'title'].iloc[0]
            recommended_books.append({'book_id': int(book_id), 'title': book_title})  # Converte book_id para int
        else:
            print(f"Index {index} fora dos limites para user_id {user_id}")  # Mensagem de erro para debug
    
    return jsonify(recommended_books)

# Rota para obter recomendações baseadas em conteúdo para um livro específico
@app.route('/content-recommendations/<int:book_id>')
def get_content_based_recommendations(book_id):
    idx = books_df.index[books_df['book_id'] == book_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Seleciona os top 5 livros mais similares
    
    book_indices = [i[0] for i in sim_scores]
    recommended_books = books_df.iloc[book_indices][['book_id', 'title']].to_dict(orient='records')
    
    return jsonify(recommended_books)

# Execução do aplicativo Flask em modo de depuração
if __name__ == '__main__':
    app.run(debug=True)
