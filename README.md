# Modelos de Regressao

Projeto de estudo em ciencia de dados focado em regressao supervisionada com Python, Pandas, Scikit-learn e boas praticas de organizacao de pipeline.

O fluxo atual usa o dataset `load_diabetes` do Scikit-learn para prever a progressao quantitativa da diabetes um ano apos a medicao inicial dos pacientes. O projeto trabalha com duas versoes da base:

- `dados/diabetes_tratados.parquet`: base numerica tratada.
- `dados/diabetes_categorizado.parquet`: base com a coluna ordinal `colesterol_hdl_cat`, usada nos pipelines com `OrdinalEncoder`.

## Descricao para o GitHub

Projeto de ciencia de dados para comparar modelos de regressao no dataset Diabetes, com EDA, versionamento de dados em Parquet, pipelines de preprocessamento, validacao cruzada e avaliacao de modelos lineares regularizados.

## Objetivo

Construir uma base solida para modelagem de regressao, passando por:

- analise exploratoria dos dados;
- tratamento e versionamento da base;
- separacao entre treino e teste;
- criacao de baseline com regressao linear;
- avaliacao por metricas de regressao;
- analise de residuos e coeficientes;
- curvas de aprendizado;
- uso de `Pipeline`, `StandardScaler`, `ColumnTransformer`, `OneHotEncoder`, `OrdinalEncoder`, transformacoes de target e validacao cruzada;
- comparacao entre `DummyRegressor`, `LinearRegression`, `Lasso`, `Ridge` e `ElasticNet`.

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
|   |-- 06_outras_transformacoes.ipynb
|   |-- 07_target_transformer.ipynb
|   |-- 08_validacao_cruzada.ipynb
|   |-- 09_dummy_regressor.ipynb
|   |-- 10. analise_complexidade.ipynb
|   `-- 11_outros_modelos.ipynb
|-- referencias/
|   `-- analise_tecnica.md
|-- relatorios/
`-- src/
    |-- config.py
    |-- graficos.py
    |-- modelos.py
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
| `07_target_transformer.ipynb` | Transformacao do alvo | Uso de `TransformedTargetRegressor` para transformar `y` durante treino e predicao |
| `08_validacao_cruzada.ipynb` | Validacao cruzada | Avaliacao com `KFold` e `cross_validate` usando metricas de regressao |
| `09_dummy_regressor.ipynb` | Baseline ingenuo | Comparacao com `DummyRegressor` para validar se modelos reais superam um baseline simples |
| `10. analise_complexidade.ipynb` | Comparacao de preprocessamentos | Comparacao entre pipelines categoricos, simples e completos |
| `11_outros_modelos.ipynb` | Modelos regularizados | Comparacao entre `LinearRegression`, `Lasso`, `Ridge` e `ElasticNet`, com variacoes de preprocessamento, transformacao do target e hiperparametros iniciais |

## Uso Correto dos Dados

Use `DADOS_TRATADOS` quando o pipeline trabalhar somente com as colunas numericas originais.

Use `DADOS_CATEGORIZADOS` quando o pipeline incluir a coluna `colesterol_hdl_cat`, por exemplo em preprocessadores com `OrdinalEncoder`:

```python
from src.config import DADOS_CATEGORIZADOS

df = pd.read_parquet(DADOS_CATEGORIZADOS)

X = df.drop(columns="target")
y = df["target"]
```

Se o preprocessador espera `colesterol_hdl_cat` e `X` foi criado a partir de `DADOS_TRATADOS`, o treino falha porque essa coluna nao existe nessa versao da base.

## Validacao Cruzada e Graficos

A funcao `treinar_e_validar_modelo_regressao` retorna o dicionario gerado por `cross_validate`. Para plotar com Seaborn, primeiro transforme esse dicionario em um DataFrame com `organiza_resultados`.

Fluxo correto:

```python
resultados = {
    nome_modelo: treinar_e_validar_modelo_regressao(X, y, **regressor)
    for nome_modelo, regressor in regressors.items()
}

df_resultados = organiza_resultados(resultados)
```

Depois use `df_resultados` no grafico:

```python
fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

comparar_metricas = [
    "time_seconds",
    "test_r2",
    "test_neg_mean_absolute_error",
    "test_neg_root_mean_squared_error",
]

nomes_metricas = [
    "Tempo(seg)",
    "R2",
    "MAE",
    "RMSE",
]

for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
    sns.boxplot(
        x="model",
        y=metrica,
        data=df_resultados,
        ax=ax,
        showmeans=True,
    )

    ax.set_title(nome)
    ax.set_ylabel(nome)
    ax.tick_params(axis="x", rotation=90)

plt.tight_layout()
```

Nao passe `resultados` diretamente para `sns.boxplot`, porque ele nao possui a coluna `model`. Essa coluna e criada pela funcao `organiza_resultados`.


## Resultado Atual

O modelo baseline de regressao linear apresenta desempenho moderado:

- `R2`: aproximadamente `0.453`;
- `MAE`: aproximadamente `42.79`;
- `MSE`: aproximadamente `2900.20`;
- `RMSE`: aproximadamente `53.85`.

Isso indica que o modelo explica cerca de 45% da variacao do target, mas ainda ha erro residual relevante. Esse resultado serve como baseline para comparacao com modelos regularizados e nao lineares.

Na etapa mais recente, o projeto amplia a comparacao para modelos regularizados:

- `Lasso`: adiciona penalizacao L1 e pode zerar coeficientes, funcionando como uma selecao implicita de variaveis.
- `Ridge`: adiciona penalizacao L2 e tende a estabilizar coeficientes em cenarios com variaveis correlacionadas.
- `ElasticNet`: combina L1 e L2, equilibrando selecao de variaveis e estabilidade.

Os modelos sao comparados com `KFold`, `cross_validate` e os mesmos indicadores principais: `R2`, `MAE`, `RMSE` e tempo de execucao. O `DummyRegressor` permanece como baseline minimo para verificar se os modelos reais capturam sinal preditivo.

## Pontos Tecnicos Importantes

- A analise exploratoria vem antes da modelagem.
- A base tratada foi salva em formato `.parquet`, mantendo um artefato de dados versionavel.
- A separacao treino/teste evita avaliar o modelo em dados ja vistos.
- `Pipeline` reduz risco de vazamento de dados e organiza o fluxo de preprocessamento e modelagem.
- `StandardScaler` torna os coeficientes da regressao linear mais comparaveis.
- `OneHotEncoder` evita interpretar categorias como valores numericos ordinais.
- `OrdinalEncoder` deve ser usado apenas em categorias com ordem real, como `colesterol_hdl_cat`.
- `ColumnTransformer` permite aplicar transformacoes diferentes por tipo de variavel.
- `cross_validate` retorna arrays por metrica; para comparacao grafica, organize os resultados em formato tabular.
- Modelos regularizados ajudam a controlar overfitting e coeficientes instaveis, mas devem ser comparados por validacao cruzada.
- Transformacoes mais complexas so devem ser mantidas quando melhoram desempenho ou interpretabilidade de forma consistente.
- As metricas de regressao devem ser calculadas com a ordem correta: `metrica(y_test, y_pred)`.

## Material de Revisao

A explicacao tecnica detalhada dos notebooks esta em:

- [`referencias/analise_tecnica.md`](referencias/analise_tecnica.md)
