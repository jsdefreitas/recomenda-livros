# recomenda-livros

Sistema de Recomendação de Livros
Este projeto implementa um sistema de recomendação de livros utilizando Flask, Pandas, e scikit-learn. O sistema oferece duas formas de recomendação:

Recomendações Colaborativas: Com base nas avaliações de outros usuários, recomenda livros para um usuário específico usando o algoritmo k-Nearest Neighbors (k-NN).
Recomendações Baseadas em Conteúdo: Com base na similaridade entre descrições de livros usando a técnica TF-IDF (Term Frequency-Inverse Document Frequency).
Configuração do Projeto
Pré-requisitos
Certifique-se de ter instalado:

Python 3.x
Flask
Pandas
scikit-learn
Instalação
Clone o repositório:

bash
Copy code
git clone https://github.com/jsdefreitas/recomenda-livros.git
cd nome-do-repositorio
Instale as dependências:

Copy code
pip install -r requirements.txt
Como Executar
Para iniciar o servidor Flask:

Copy code
python app.py
O sistema estará disponível em http://localhost:5000/.

Endpoints
Recomendações Colaborativas:

sql
Copy code
GET /recommendations/<int:user_id>
Retorna até 5 recomendações de livros para o usuário especificado.

Recomendações Baseadas em Conteúdo:

sql
Copy code
GET /content-recommendations/<int:book_id>
Retorna até 5 livros mais similares ao livro especificado.

Estrutura do Projeto
app.py: Contém a lógica principal do servidor Flask e os endpoints de API.
templates/index.html: Página HTML para interface de usuário.
user_reading_history.csv: Histórico de leitura dos usuários.
book_metadata.csv: Metadados dos livros.
Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues para sugestões de melhorias ou novos recursos.
