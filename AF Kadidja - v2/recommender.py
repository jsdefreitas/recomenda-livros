import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Definir caminhos dos arquivos corretamente
user_reading_history_path = r'C:\Users\jfreitas\Downloads\AF Kadidja - v2\user_reading_history.csv'
book_metadata_path = r'C:\Users\jfreitas\Downloads\AF Kadidja - v2\book_metadata.csv'

# Verificar se os arquivos existem
if not os.path.exists(user_reading_history_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {user_reading_history_path}")

if not os.path.exists(book_metadata_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {book_metadata_path}")

# Carregar dados de leitura do arquivo CSV
df = pd.read_csv(user_reading_history_path)

# Exibir os dados carregados (opcional, apenas para verificação)
print("Dados carregados:")
print(df)

# Criar matriz de usuários e livros
user_book_matrix = df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

# Exibir a matriz de usuários e livros
print("\nMatriz de usuários e livros:")
print(user_book_matrix)

# Converter matriz para formato esparso
user_book_matrix_sparse = csr_matrix(user_book_matrix.values)

# Modelo de KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_book_matrix_sparse)

print("Modelo KNN treinado com sucesso.")

# Carregar metadados dos livros
books_df = pd.read_csv(book_metadata_path)

# Exibir os dados carregados
print("\nMetadados dos livros carregados:")
print(books_df)

# Treinar o modelo de TF-IDF com base nas descrições dos livros
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['description'])

# Calcular a similaridade de cosseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print("Modelo TF-IDF treinado com sucesso.")

# Função de recomendação colaborativa
def recommend_books(user_id, n_recommendations=5):
    user_index = user_book_matrix.index.tolist().index(user_id)
    n_users = user_book_matrix.shape[0]
    if n_recommendations > n_users - 1:
        n_recommendations = n_users - 1  # Limitar o número de recomendações ao máximo possível
    
    distances, indices = model_knn.kneighbors(user_book_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=n_recommendations+1)
    
    recommended_books = []
    for i in range(1, len(distances.flatten())):
        recommended_books.append(user_book_matrix.columns[indices.flatten()[i]])
    
    return recommended_books

# Testar a função de recomendação colaborativa para um usuário
print(f"\nRecomendações colaborativas para o usuário 1: {recommend_books(1)}")

# Função de recomendação baseada em conteúdo
def recommend_books_based_on_content(book_id, n_recommendations=5):
    idx = books_df.index[books_df['book_id'] == book_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations+1]
    
    book_indices = [i[0] for i in sim_scores]
    return books_df.iloc[book_indices]['title'].tolist()

# Testar a função de recomendação baseada em conteúdo para um livro
print(f"\nRecomendações baseadas em conteúdo para o livro 101: {recommend_books_based_on_content(101)}")
