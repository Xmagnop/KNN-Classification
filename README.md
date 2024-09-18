# python-repository-template

A Python repository template to facilitate getting your projects started and organized.

# If you use Windows, use chocolatey for installing things

- [chocolatey installation guide](https://chocolatey.org/install)

# Use pyenv for Python version management

- [pyenv installation guide](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)

```bash
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile
exit
```

In another shell:

```bash
pyenv update
pyenv install 3.9.13
pyenv rehash
pyenv global 3.9.13
exit
```

# Use `make` for simplifing commands and making it explicit how to run your code

- [make documentation](https://www.gnu.org/software/make/manual/make.html)

# Use poetry for managing Python dependencies

[Poetry](https://python-poetry.org/docs/basic-usage/) is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Poetry offers a lockfile to ensure repeatable installs, and can build your project for distribution.

## Basic commands:

- Add new dependency: `poetry add <package>`
- Install dependencies: `poetry install`
- Update dependencies: `poetry update`
- Remove dependencies: `poetry remove <package>`
- Run a command in the virtual environment: `poetry run <command>`
- Run python in the virtual environment: `poetry run python <command>`

# Make sure to use the Makefile to facilitate the usage of your repository

Anyone that clones your repository should be able to relatively easily run your code with just a few commands. The Makefile should contain the following commands:

```bash
make install
make run
```

# Use pre-commit for running checks before committing

[pre-commit](https://pre-commit.com/) is a framework for managing and maintaining multi-language pre-commit hooks. It is a client-side hook manager that can be used to automate checks before committing code. It is recommended to use pre-commit to ensure code formatting, among other things.


# Classificação de Dígitos Usando KNN com PCA e LDA

Este projeto implementa uma solução para classificar dígitos do conjunto de dados MNIST em números pares ou ímpares utilizando o algoritmo de K-Nearest Neighbors (KNN). O código faz uso de duas técnicas de redução de dimensionalidade: Análise de Componentes Principais (PCA) e Análise Discriminante Linear (LDA). Além disso, utiliza a técnica SMOTE para balanceamento de classes em um dos métodos.

## Estrutura Geral do Projeto

- **main()**: Função principal que carrega o conjunto de dados e executa os métodos de classificação KNN com PCA e LDA.

- **PCA_KNN(digits)**: Implementa o algoritmo KNN após a redução de dimensionalidade e extração de características usando PCA.

- **LDA_KNN(digits)**: Implementa o algoritmo KNN após a redução de dimensionalidade e extração de características usando LDA e realiza balanceamento de classes com SMOTE.

- **plot_digits(images, labels, predictions)**: Função auxiliar para visualizar os dígitos e comparar os rótulos reais com as previsões feitas pelo KNN.

## Funcionamento dos Algoritmos

### Carregamento dos Dados e Classificação Binária

O conjunto de dados utilizado é o MNIST, que contém imagens de dígitos escritos à mão (0-9). Cada imagem tem resolução de 8x8 pixels, sendo representada como uma matriz de 64 valores. Neste projeto, o problema é transformado em uma classificação binária, onde os dígitos são categorizados como pares (0) ou ímpares (1).

### Algoritmo KNN

O KNN (K-Nearest Neighbors) é um algoritmo de classificação que funciona da seguinte forma:

- Dado um ponto a ser classificado, o algoritmo calcula a distância entre este ponto e todos os pontos de treinamento.

- Em seguida, ele identifica os K pontos mais próximos (vizinhos) desse ponto.

- Finalmente, a classe que aparece com mais frequência entre os vizinhos é atribuída ao ponto.

Neste projeto, o KNN é aplicado com duas diferentes técnicas de redução de dimensionalidade: PCA e LDA.

### Extração de Caraterísticas e Redução de Dimensionalidade

#### PCA (Principal Component Analysis)

O PCA é uma técnica de extração de caracteristicas e redução de dimensionalidade não supervisionada. Isso significa que ele não leva em conta os rótulos (ou classes) durante o processo de transformação dos dados. O objetivo do PCA é encontrar uma nova base ortogonal (chamada de componentes principais) na qual os dados possam ser projetados, de modo que a maior parte da variância dos dados seja capturada nas primeiras direções principais.

Os principais passos do PCA são:

- **Cálculo da Covariância**: O primeiro passo é normalizar os dados e calcular a matriz de covariância, que mede como as diferentes variáveis se correlacionam entre si.

- **Autovalores e Autovetores**: A partir da matriz de covariância, o PCA calcula os autovalores e autovetores. Os autovetores representam as direções principais (componentes principais), enquanto os autovalores indicam a quantidade de variância capturada por cada um destes componentes.

- **Projeção dos Dados**: Finalmente, os dados são projetados no espaço definido pelos autovetores. As primeiras componentes principais são aquelas que capturam a maior parte da variância.

Vantagens do PCA:

- **Redução de Ruído**: Ao eliminar dimensões que não carregam informações relevantes para classificação (i.e., ruído), o PCA pode ajudar a melhorar a precisão do modelo.

- **Desempenho Computacional**: Reduzir o número de dimensões torna o treinamento do modelo mais rápido e eficiente, principalmente em algoritmos como o KNN, que precisam calcular distâncias entre os pontos.

#### LDA (Linear Discriminant Analysis)

O LDA é uma técnica de extração de características e redução de dimensionalidade supervisionada, o que significa que ele considera os rótulos das classes ao reduzir as dimensões. O objetivo principal do LDA é maximizar a separação entre as classes projetando os dados em um novo espaço onde a separação entre as classes seja a maior possível.

Os principais passos do LDA são:

- **Cálculo das Médias**: O LDA calcula as médias das classes no espaço de entrada.

- **Maximização da Separação entre Classes**: Ele encontra a combinação linear de características que maximiza a distância entre as médias das classes e minimiza a dispersão dentro das classes.

- **Projeção dos Dados**: Os dados são projetados no espaço onde a separação entre as classes é maximizada.

Vantagens do LDA:

- **Melhor Separação de Classes**: Como o LDA leva em conta os rótulos das classes durante a redução de dimensionalidade, ele pode ser mais eficaz do que o PCA em cenários onde as classes são bem separáveis linearmente.

- **Facilidade de Interpretação**: Como o LDA maximiza a distância entre classes, ele tende a produzir resultados que são mais facilmente interpretáveis.

#### Balanceamento de Dados com SMOTE

Em problemas de classificação binária, pode haver um desequilíbrio entre o número de exemplos em cada classe. O SMOTE (Synthetic Minority Over-sampling Technique) é utilizado para criar exemplos sintéticos da classe minoritária, ajudando a equilibrar o conjunto de treinamento.

Vantagens do SMOTE:

- **Aumento da Precisão**: O balanceamento das classes evita que o modelo fique enviesado para a classe majoritária.

- **Melhor Generalização**: O SMOTE ajuda o modelo a aprender a reconhecer padrões da classe minoritária, o que melhora a sua capacidade de generalização.