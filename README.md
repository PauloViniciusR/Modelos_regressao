# Modelos de Regressao

Projeto de estudo em ciencia de dados focado em regressao supervisionada com Python, Pandas, Scikit-learn e boas praticas de organizacao de pipeline.

O fluxo atual usa o dataset `load_diabetes` do Scikit-learn para prever a progressao quantitativa da diabetes um ano apos a medicao inicial dos pacientes.

## Objetivo

Construir uma base solida para modelagem de regressao, passando por:

- analise exploratoria dos dados;
- tratamento e versionamento da base;
- separacao entre treino e teste;
- criacao de baseline com regressao linear;
- avaliacao por metricas de regressao;
- analise de residuos e coeficientes;
- curvas de aprendizado;
- uso de `Pipeline`, `StandardScaler`, `ColumnTransformer` e `OneHotEncoder`.

## Estrutura do Projeto

```text
.
|-- dados/
|   |-- diabetes_categorizado.parquet
|   `-- diabetes_tratados.parquet
|-- modelos/
|-- notebooks/
|   |-- 01_EDA.ipynb
|   |-- 02_regressao_linear.ipynb
|   |-- 03_curva_aprendizado.ipynb
|   |-- 04_introducao_a_pipelines.ipynb
|   |-- 05_one_hot.ipynb
|   `-- 06_outras_transformacoes.ipynb
|-- referencias/
|   |-- 01_dicionario_de_dados.md
|   `-- analise_tecnica.md
|-- relatorios/
`-- src/
    |-- config.py
    |-- graficos.py
    `-- utils.py
```

## Notebooks

| Notebook | Tema principal | Conteudo |
|---|---|---|
| `01_EDA.ipynb` | Analise exploratoria | Carregamento do dataset, renomeacao de colunas, estatisticas descritivas, correlacao e exportacao da base tratada |
| `02_regressao_linear.ipynb` | Baseline | Treino de `LinearRegression`, avaliacao com `MAE`, `MSE`, `RMSE`, `R2`, coeficientes e residuos |
| `03_curva_aprendizado.ipynb` | Diagnostico | Uso de curva de aprendizado para avaliar comportamento do modelo conforme o volume de treino |
| `04_introducao_a_pipelines.ipynb` | Pipeline | Encapsulamento de `StandardScaler` e `LinearRegression` em um `Pipeline` |
| `05_one_hot.ipynb` | Variaveis categoricas | Uso de `ColumnTransformer` e `OneHotEncoder` para tratar a coluna `sexo` corretamente |
| `06_outras_transformacoes.ipynb` | Transformacoes mistas | Uso de `PowerTransformer`, `OrdinalEncoder`, `OneHotEncoder` e `StandardScaler` no mesmo pipeline |


## Resultado Atual

O modelo baseline de regressao linear apresenta desempenho moderado:

- `R2`: aproximadamente `0.453`;
- `MAE`: aproximadamente `42.79`;
- `MSE`: aproximadamente `2900.20`;
- `RMSE`: aproximadamente `53.85`.

Isso indica que o modelo explica cerca de 45% da variacao do target, mas ainda ha erro residual relevante. Esse resultado serve como baseline para comparacao com modelos regularizados e nao lineares.

## Pontos Tecnicos Importantes

- A analise exploratoria vem antes da modelagem.
- A base tratada foi salva em formato `.parquet`, mantendo um artefato de dados versionavel.
- A separacao treino/teste evita avaliar o modelo em dados ja vistos.
- `Pipeline` reduz risco de vazamento de dados e organiza o fluxo de preprocessamento e modelagem.
- `StandardScaler` torna os coeficientes da regressao linear mais comparaveis.
- `OneHotEncoder` evita interpretar categorias como valores numericos ordinais.
- `ColumnTransformer` permite aplicar transformacoes diferentes por tipo de variavel.
- As metricas de regressao devem ser calculadas com a ordem correta: `metrica(y_test, y_pred)`.

## Material de Revisao

A explicacao tecnica detalhada dos notebooks esta em:

- [`referencias/analise_tecnica.md`](referencias/analise_tecnica.md)

## Proximos Passos

- Comparar modelos regularizados: `Ridge`, `Lasso` e `ElasticNet`.
- Usar validacao cruzada para obter avaliacao mais estavel.
- Testar modelos nao lineares, como `RandomForestRegressor` e `GradientBoostingRegressor`.
- Salvar modelos treinados em `modelos/`.
- Gerar relatorios e imagens em `relatorios/`.
