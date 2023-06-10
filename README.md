# Modelo de PNL 

# Tecnologias utilizadas:
- python notebook
- nltk
- keras
- scikit-learn
- TensorFlow

# Estrutura de pastas:
```
├── machine-learning
│    ├── base de dados
│        ├── imdb-reviews-pt-br
│    ├── dadosTratados
│        ├── resultadoTratamento.csv
│    ├── modelo
│        ├── modelo_Bert.h5
│    ├── notebook de treino do modelo
│        ├── ModeloPLNMindShield.ipynb
│    ├── README.md
```

# Objetivo do modelo: 

O objetivo do meu modelo é utilizar de PLN, Processamento de Linguagem Natural,  para a detecção de sentimentos ruins nos comentários das crianças. De forma a ser possível alertar quando há a necessidade de tratamento psicológico.

Ao analisar os comentários das crianças, o modelo é capaz de identificar se as respostas são positivas ou negativas. Essa análise permite que sejam identificados  problemas emocionais e traumas que requerem atenção e cuidado, sendo possível gerar alerta para profissionais especializados.

Com essa detecção precoce, os profissionais de saúde e educadores podem ser alertados sobre a possível necessidade de intervenção psicológica, possibilitando um suporte adequado e evitando que os sentimentos da criança e do adolecente saiam de controle.

A utilização da IA e do PLN nesse contexto proporciona uma ferramenta valiosa para auxiliar na identificação de problemas emocionais nas crianças, contribuindo para uma abordagem mais preventiva, e  mais eficaz na prevenção de atos violentos e no cuidado da saúde mental das crianças.

# Rodar a aplicação
Para rodar o notebook de treinamento,caso requisitado pelo sistema devido às tecnologias utilizadas, as seguintes dependências devem ser instaladas:

- unidecode
- sentence-transformers
- spacy
- nltk
- keras
- scikit-learn
- matplotlib
- numpy
- imbalanced-learn
- tensorflow
- joblib
- transformers


Também é necessário substituir o caminho de importação do dataset para o caminho no qual ele está em sua máquina.

```phyton
df = pd.read_csv("/content/drive/MyDrive/Hackas/imdb-reviews-pt-br.csv")
```

# Pre-processamento:
Etapas iniciais de preparação e limpeza dos dados textuais antes de serem usados nos modelos, realizadas para remover ruído e melhorar a consistência dos dados. Para o pré processamento foram realizados cinco tratamentos:

## Remoção de acentos: 
Esse tratamento visa a remoção dos acentos de todas as palavras de forma a manter uma padronização entre elas.
## Tratamente de letras Maiusculas: 
Esse tratamento transforma todas as letras em minúsculas de forma a manter a padronização.
## Lematização :  
É um processo de redução de palavras a sua forma base, levando em conta o contexto e a estrutura gramatical, foi utilizado para agrupar as palavras de mesmo significado.
## Stopwords:  
Stopwords são palavras sem muito valor para o modelo como "o", "a", "de", sendo removidas por possuírem pouca relevância.
##Tokenização: 
Esse tratamento tem como objetivo transformar todas as palavras em tokens, de forma a serem mais adequadas para o modelo as entender.

# Vetorização:
Processo de converter dados textuais em representações numéricas para que possam ser entendidos pelos algoritmos de machine learning. Foram realizados 3 tipos de vetorização: 

## Bag of Words: 
Técnica de vetorização que considera o texto como uma lista de palavras, criando vetores com a contagem de quantas vezes cada palavra aparece.
## Transformer : 
O Transformer é um modelo de embedding l que utiliza mecanismos de atenção para capturar relações entre palavras, e gerar vetores com base nessa relação.
## Bert
Baseado em Transformers também captura informações contextuais de palavras em textos, mas dando uma ênfase ainda maior no contexto.

# Tipos de modelo:
Os tipos de vetorização foram utilizados nos seguintes modelos:

## Naive Bayes
Naive Bayes é um algoritmo de classificação com base em probabilidades, baseado no teorema de Bayes.

## Rede Neural
Modelo de aprendizado de máquina composto por diversas camadas de neurônios artificiais , capaz de aprender com dados e realizar classificações com base nesses dados.

# Resultado dos modelos:

## Naive Bayes com bag of words:

Acurácia: 0.900136

Revocação: 0.9000

Matriz: 

![image](https://github.com/Mind-Shield/machine-learning/assets/99202408/46339610-8cf8-4750-9704-a9ab5cce7743)

## Naive Bayes com Transformers

Acurácia: 0.772702

Revocação: 0.7720

Matriz: 

![image](https://github.com/Mind-Shield/machine-learning/assets/99202408/36bdadbb-dc58-4ee7-b185-251ea0311b4b)


## Rede Neural com Transformers	

Acurácia: 0.811323

Revocação: 0.8112

Matriz: 

![image](https://github.com/Mind-Shield/machine-learning/assets/99202408/ad00e9ee-5be8-4e44-b955-13773c9df7de)


## Rede Neural com Bert	

Acurácia: 0.813862

Revocação: 0.7952

Matriz: 

![image](https://github.com/Mind-Shield/machine-learning/assets/99202408/e200e5c8-376d-4b59-a9a7-d9fcb5e7733b)

## License

Distributed under the MIT License. See [License](https://github.com/Mind-Shield/machine-learning/blob/main/LICENSE) for more information.

