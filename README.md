# FundusDetectionTCC

Este é um projeto para detecção de catarata usando redes neurais com base nos modelos VGG, ResNet, MobileNet LLMs (Vision Transformer, Compact Convolution Transformes) e LLMs modificados. O objetivo é classificar imagens de fundo de olho para detectar sinais de catarata.

## Requisitos

Para rodar este projeto, você precisará de:

- Python 3.x
- Bibliotecas listadas em `requirements.txt`

Você pode instalar as dependências com o seguinte comando:

```bash
pip install -r requirements.txt
```

## Dataset

O dataset necessário para este projeto não está incluído diretamente. Você pode baixá-lo a partir do seguinte link:

[Baixar Dataset no Kaggle](https://www.kaggle.com/code/ragilhadip/eye-disease-classification-with-grey-level-co-occu/)

Após o download, extraia o dataset na pasta principal do projeto (onde estão localizados os arquivos `README.md` e `main.py`).

## Estrutura do Projeto

O projeto possui a seguinte estrutura:

```
FundusDetectionTCC-main/
│
├── .gitignore
├── README.md
├── main.py
├── requirements.txt
├── src/                # Código-fonte
│   ├── data_augmentation.py       
│   ├── dataset_gen_load.py       
│   ├── gen_models.py            
│   ├── partitions_dataset.py    
│   ├── preprocessamento.py     
│   ├── utils_dataset.py        
│   ├── utils_model.py          
│   └── utils_plot.py           
└── treinar.ipynb       # Notebook para treinamento do modelo
└── Artigo Classificacao Felipe TCC.pdf # Artigo utilizando o projeto
```

## Descrição dos Arquivos

- **`.gitignore`**: Arquivo de configuração para ignorar arquivos específicos que não devem ser versionados pelo Git.
- **`README.md`**: Documento explicativo sobre o projeto, como este.
- **`main.py`**: Script principal para execução do projeto.
- **`requirements.txt`**: Arquivo contendo todas as dependências do projeto.
- **`src/`**: Pasta com o código-fonte do projeto.
    - **`data_augmentation.py`**: Implementa técnicas de aumento de dados (Data Augmentation) para enriquecer o dataset e evitar overfitting.
    - **`dataset_gen_load.py`**: Contém funções para gerar e carregar o dataset necessário para treinamento e teste dos modelos.
    - **`gen_models.py`**: Define as arquiteturas dos modelos de redes neurais como VGG, ResNet e MobileNet.
    - **`partitions_dataset.py`**: Responsável por dividir o dataset em conjuntos de treino, validação e teste.
    - **`preprocessamento.py`**: Realiza o pré-processamento das imagens para adequá-las à entrada dos modelos, como redimensionamento e normalização.
    - **`utils_dataset.py`**: Funções auxiliares para manipulação de dados, como leitura e organização das imagens.
    - **`utils_model.py`**: Funções auxiliares para manipulação dos modelos, incluindo funções de treinamento, avaliação e salvamento de modelos.
    - **`utils_plot.py`**: Funções para visualização e plotagem de gráficos e métricas de desempenho.
- **`treinar.ipynb`**: Notebook Jupyter para treinamento do modelo.
- **`Artigo Classificacao Felipe TCC.pdf`**: Artigo que descreve o projeto e seus resultados.


## Artigo

Você pode acessar o artigo completo que descreve este projeto e seus resultados no arquivo [Artigo Classificacao Felipe TCC.pdf](Artigo%20Classificacao%20Felipe%20TCC.pdf).

## Como Executar

1. Instale as dependências com `pip install -r requirements.txt`.
2. Baixe o dataset conforme instruído.
3. Execute o script principal:

```bash
python main.py
```

Ou, se preferir, execute o notebook `treinar.ipynb` para treinar o modelo.
