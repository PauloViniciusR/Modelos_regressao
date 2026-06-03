# Revisao Tecnica dos Notebooks de Regressao

Este documento resume os principais conceitos, decisoes tecnicas e resultados vistos ate agora nos notebooks do projeto.

## Visao Geral

O projeto esta estruturado como um fluxo inicial de ciencia de dados para regressao supervisionada. A base usada e o dataset `load_diabetes`, disponivel no Scikit-learn.

O objetivo do problema e prever uma medida quantitativa de progressao da diabetes um ano apos a coleta inicial de variaveis clinicas dos pacientes.

A sequencia dos notebooks segue uma ordem adequada:

1. analisar os dados;
2. tratar e salvar a base;
3. criar um modelo baseline;
4. avaliar desempenho;
5. diagnosticar aprendizado;
6. organizar preprocessamento e modelo em pipelines;
7. tratar variaveis categoricas corretamente.
8. aplicar transformacoes em features e target;
9. avaliar modelos com validacao cruzada;
10. comparar modelos e preprocessamentos por metricas.

## 01 - Analise Exploratoria dos Dados

Notebook: `notebooks/01_EDA.ipynb`

### O que foi feito

Foi carregado o dataset de diabetes do Scikit-learn:

```python
from sklearn.datasets import load_diabetes

dados = load_diabetes(as_frame=True, scaled=False)
```

A base possui:

- `442` linhas;
- `10` variaveis explicativas;
- `1` variavel alvo, chamada `target`.

As colunas originais foram renomeadas para nomes mais interpretaveis em portugues:

| Coluna original | Nome usado no projeto |
|---|---|
| `age` | `idade` |
| `sex` | `sexo` |
| `bmi` | `imc` |
| `bp` | `pressao_media` |
| `s1` | `colesterol_total` |
| `s2` | `ldl` |
| `s3` | `hdl` |
| `s4` | `colesterol_hdl` |
| `s5` | `triglicerides` |
| `s6` | `glicose` |
| `target` | `target` |

### Analises realizadas

Foram avaliados:

- primeiras linhas com `head`;
- estrutura da base com `info`;
- estatisticas descritivas com `describe`;
- distribuicoes das variaveis;
- matriz de correlacao;
- mapa de calor com Seaborn.

Um ponto importante e que a base nao possui valores nulos. Isso simplifica a primeira etapa de modelagem, pois nao foi necessario aplicar imputacao.

### Correlacao

A matriz de correlacao indicou algumas variaveis com relacao mais forte com o `target`, especialmente:

- `imc`;
- `pressao_media`;
- `triglicerides`;
- `colesterol_hdl`.

Essa analise nao prova causalidade. Ela apenas indica associacao linear entre as variaveis e ajuda a orientar a interpretacao inicial.

### Otimizacao de memoria

Tambem foi feito ajuste dos tipos numericos com `downcast`.

Antes:

```text
memory usage: 38.1 KB
```

Depois:

```text
memory usage: 13.5 KB
```

Essa pratica e util principalmente em bases maiores, porque reduz custo de memoria sem alterar a estrutura analitica da base.

### Exportacao da base tratada

A base final foi salva em:

```text
dados/diabetes_tratados.parquet
```

Isso cria uma separacao importante entre:

- dados brutos carregados do Scikit-learn;
- dados tratados usados nos proximos notebooks.

Nos notebooks com categorizacao de `colesterol_hdl`, tambem e usada a base:

```text
dados/diabetes_categorizado.parquet
```

Essa segunda versao contem a coluna `colesterol_hdl_cat`, necessaria para os pipelines com `OrdinalEncoder`.

## 02 - Regressao Linear

Notebook: `notebooks/02_regressao_linear.ipynb`

### O que foi feito

Foi criado o primeiro modelo baseline com `LinearRegression`.

O dataset foi separado em:

```python
X = df.drop(columns="target")
y = df["target"]
```

Depois, foi feita a divisao entre treino e teste:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Essa divisao e fundamental para medir a capacidade de generalizacao do modelo. O modelo aprende com os dados de treino e e avaliado em dados que nao foram usados no ajuste.

### Modelo

O modelo usado foi:

```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

Esse modelo tenta ajustar uma relacao linear entre as variaveis explicativas e o target.

### Metricas

As principais metricas avaliadas foram:

| Metrica | Valor aproximado | Interpretacao |
|---|---:|---|
| `R2` | `0.453` | O modelo explica cerca de 45% da variacao do target |
| `MAE` | `42.79` | Erro absoluto medio em unidades do target |
| `MSE` | `2900.20` | Penaliza erros maiores por elevar o erro ao quadrado |
| `RMSE` | `53.85` | Erro medio na escala original do target |

O resultado e aceitavel como baseline, mas ainda existe bastante erro residual.

### Coeficientes

Os coeficientes indicam o impacto estimado de cada variavel na predicao, mantendo as demais constantes.

Alguns coeficientes observados:

| Variavel | Coeficiente aproximado |
|---|---:|
| `triglicerides` | `67.11` |
| `sexo` | `-23.06` |
| `colesterol_hdl` | `10.16` |
| `imc` | `5.85` |
| `colesterol_total` | `-1.28` |

Como as variaveis ainda nao estavam padronizadas nesse notebook, a comparacao direta entre magnitudes dos coeficientes deve ser feita com cuidado. Variaveis em escalas diferentes podem gerar coeficientes numericamente diferentes mesmo quando sua importancia real nao e proporcional a esse valor.

## 03 - Curva de Aprendizado

Notebook: `notebooks/03_curva_aprendizado.ipynb`

### Objetivo

A curva de aprendizado foi usada para avaliar como o desempenho do modelo muda conforme a quantidade de dados de treino aumenta.

Foi usado:

```python
from sklearn.model_selection import learning_curve, LearningCurveDisplay
```

### Conceitos principais

A curva compara desempenho em:

- treino;
- validacao.

Ela ajuda a diagnosticar tres situacoes comuns.

### Overfitting

Ocorre quando o modelo aprende muito bem os dados de treino, mas nao generaliza bem para dados novos.

Sinais comuns:

- desempenho alto no treino;
- desempenho baixo na validacao;
- grande distancia entre as curvas.

### Underfitting

Ocorre quando o modelo e simples demais ou nao consegue capturar os padroes relevantes dos dados.

Sinais comuns:

- desempenho ruim no treino;
- desempenho ruim na validacao;
- curvas proximas, mas em nivel baixo.

### Boa generalizacao

Ocorre quando as curvas de treino e validacao ficam proximas e com bom desempenho.

Esse e o comportamento desejado.

## 04 - Introducao a Pipelines

Notebook: `notebooks/04_introducao_a_pipelines.ipynb`

### O que foi feito

Foi introduzido o uso de `Pipeline` do Scikit-learn.

O pipeline criado combinou:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("reg", LinearRegression())
])
```

### Por que usar pipeline

Pipelines sao importantes porque:

- organizam o fluxo de preprocessamento e modelagem;
- reduzem repeticao de codigo;
- evitam vazamento de dados;
- facilitam validacao cruzada;
- facilitam busca de hiperparametros com `GridSearchCV` ou `RandomizedSearchCV`;
- tornam o modelo final mais facil de salvar e reutilizar.

### StandardScaler

O `StandardScaler` transforma as variaveis para media zero e desvio padrao um.

Isso e especialmente importante em modelos sensiveis a escala, como:

- regressao linear com regularizacao;
- KNN;
- SVM;
- modelos baseados em distancia;
- redes neurais.

Na regressao linear simples, a padronizacao nao deve mudar as predicoes de forma relevante, mas torna os coeficientes mais comparaveis.

### Resultado

As metricas de erro permaneceram praticamente iguais ao notebook anterior:

| Metrica | Valor aproximado |
|---|---:|
| `MAE` | `42.79` |
| `MSE` | `2900.19` |
| `RMSE` | `53.85` |

### Ponto de atencao corrigido

As metricas devem receber primeiro os valores reais e depois as predicoes:

```python
r2_score(y_test, y_pred)
```

No notebook de pipeline, essa ordem foi corrigida para `MAE`, `MSE`, `RMSE`, `R2` e para o grafico de residuos. O `R2` correto fica proximo de:

```text
0.453
```

Essa observacao e importante porque algumas metricas sao simetricas, mas o `R2` nao deve ser tratado assim.

## 05 - One-Hot Encoding e ColumnTransformer

Notebook: `notebooks/05_one_hot.ipynb`

### Problema tratado

A coluna `sexo` aparece numericamente como `1` e `2`, mas conceitualmente ela representa uma categoria.

Se essa coluna for usada diretamente como numero, o modelo pode interpretar uma relacao ordinal inexistente. Por exemplo, pode assumir que `2` e maior que `1` em sentido quantitativo, quando na verdade sao apenas categorias.

### Conversao para categoria

Foi feita a conversao:

```python
df["sexo"] = df["sexo"].astype("category")
```

### ColumnTransformer

Foi usado `ColumnTransformer` para aplicar transformacoes diferentes em grupos diferentes de colunas:

```python
ColumnTransformer([
    ("numeric", StandardScaler(), colunas_numericas),
    ("categoric", OneHotEncoder(drop="if_binary"), ["sexo"])
])
```

Isso e uma boa pratica porque dados reais normalmente possuem diferentes tipos de variaveis:

- numericas continuas;
- numericas discretas;
- categoricas;
- datas;
- textos.

Cada grupo exige tratamentos diferentes.

### OneHotEncoder

O `OneHotEncoder` transforma categorias em colunas binarias.

Como `sexo` e uma variavel binaria, foi usado:

```python
OneHotEncoder(drop="if_binary")
```

Isso evita criar colunas redundantes para uma variavel com apenas duas categorias.

### Pipeline final

O fluxo final combinou:

1. preprocessamento numerico com `StandardScaler`;
2. preprocessamento categorico com `OneHotEncoder`;
3. modelo `LinearRegression`.

Esse e um formato muito mais proximo de um pipeline profissional de machine learning.

### Resultado

As metricas ficaram praticamente iguais ao baseline:

| Metrica | Valor aproximado |
|---|---:|
| `R2` | `0.453` |
| `MAE` | `42.79` |
| `MSE` | `2900.19` |
| `RMSE` | `53.85` |

Isso mostra que o tratamento categorico nao melhorou necessariamente o desempenho nesse dataset, mas melhorou a corretude tecnica do pipeline.

## 06 - Outras Transformacoes

Notebook: `notebooks/06_outras_transformacoes.ipynb`

### O que foi feito

Esse notebook evolui o preprocessamento para um fluxo com transformacoes diferentes por grupo de variaveis.

Foram usadas:

- `PowerTransformer(method="box-cox")` em `imc`, `ldl`, `hdl` e `colesterol_total`;
- `StandardScaler` em `idade`, `pressao_media`, `triglicerides` e `glicose`;
- `OrdinalEncoder` na nova variavel `colesterol_hdl_cat`;
- `OneHotEncoder(drop="if_binary")` em `sexo`;
- `LinearRegression` como estimador final.

### Categorizacao de colesterol_hdl

A variavel `colesterol_hdl` foi arredondada, convertida para inteiro e depois categorizada em faixas:

```python
pd.cut(
    df["colesterol_hdl"],
    bins=[2, 4, 6, 10],
    labels=["2-3", "4-5", "6+"],
    right=False,
)
```

Como essas faixas possuem ordem natural, o uso de `OrdinalEncoder` faz sentido nesse caso.

### Resultado

O modelo com transformacoes mistas apresentou:

| Metrica | Valor aproximado |
|---|---:|
| `R2` | `0.442` |
| `MAE` | `43.64` |
| `MSE` | `2957.29` |
| `RMSE` | `54.38` |

O desempenho ficou um pouco pior que o baseline linear anterior. Isso e um ponto importante: nem toda transformacao melhora o modelo. Transformacoes devem ser avaliadas por metrica, validacao e coerencia com o problema.

## 07 - Transformacao do Target

Notebook: `notebooks/07_target_transformer.ipynb`

### O que foi feito

Esse notebook introduz o uso de `TransformedTargetRegressor`, que permite aplicar uma transformacao em `y` durante o treino e reverter essa transformacao durante a predicao.

O fluxo fica conceitualmente assim:

1. transforma `y` no treinamento;
2. ajusta o modelo usando o target transformado;
3. gera predicoes;
4. aplica a transformacao inversa para voltar a escala original do target.

### Por que usar

Transformar o target pode ajudar quando `y` possui assimetria, caudas longas ou distribuicao pouco adequada para modelos lineares.

No projeto, essa tecnica foi combinada com:

- `Pipeline`;
- `ColumnTransformer`;
- transformacoes numericas e categoricas;
- `LinearRegression`.

### Ponto tecnico

O target deve ser transformado dentro do processo de validacao ou treino, nao manualmente antes da separacao dos dados. O `TransformedTargetRegressor` ajuda a manter esse fluxo mais seguro e reproduzivel.

## 08 - Validacao Cruzada

Notebook: `notebooks/08_validacao_cruzada.ipynb`

### O que foi feito

Foi introduzida a avaliacao com `KFold` e `cross_validate`.

Em vez de depender de uma unica divisao treino/teste, a validacao cruzada avalia o modelo em diferentes particoes dos dados. Isso gera uma estimativa mais estavel de desempenho.

Exemplo do conceito:

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    model,
    X,
    y,
    cv=kf,
    scoring=[
        "r2",
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
    ],
)
```

### Interpretacao das metricas negativas

No Scikit-learn, metricas de erro usadas em `scoring` aparecem com sinal negativo quando seguem a convencao de que "maior e melhor".

Por isso:

- `neg_mean_absolute_error` representa o `MAE` com sinal negativo;
- `neg_root_mean_squared_error` representa o `RMSE` com sinal negativo.

Para interpretar como erro comum, pode-se multiplicar por `-1`.

## 09 - Dummy Regressor

Notebook: `notebooks/09_dummy_regressor.ipynb`

### O que foi feito

Foi usado `DummyRegressor` como baseline ingenuo.

Esse modelo nao aprende relacoes reais entre `X` e `y`. Com `strategy="mean"`, por exemplo, ele sempre prediz a media do target observada no treino.

### Por que isso importa

Um modelo real precisa superar um baseline simples. Se `LinearRegression`, modelos regularizados ou modelos nao lineares nao superarem o `DummyRegressor`, existe algum problema no pipeline, nas features, na avaliacao ou na propria capacidade preditiva dos dados.

Esse passo melhora a qualidade da analise porque evita comparar modelos apenas entre si sem uma referencia minima.

## 10 - Analise de Complexidade

Notebook: `notebooks/10. analise_complexidade.ipynb`

### O que foi feito

Esse notebook compara diferentes niveis de preprocessamento:

- preprocessamento categorico;
- preprocessamento simples;
- preprocessamento completo.

Como o preprocessamento completo usa `colesterol_hdl_cat`, o notebook deve carregar:

```python
from src.config import DADOS_CATEGORIZADOS

df = pd.read_parquet(DADOS_CATEGORIZADOS)
```

Se for usado `DADOS_TRATADOS`, a coluna `colesterol_hdl_cat` nao existe e o `ColumnTransformer` falha.

### Erro comum corrigido

O objeto `resultados` retornado pela validacao cruzada e um dicionario. Ele nao deve ser passado diretamente para `sns.boxplot`, porque o Seaborn espera um DataFrame com colunas como `model`, `test_r2` e `time_seconds`.

Fluxo correto:

```python
resultados = {
    nome_modelo: treinar_e_validar_modelo_regressao(X, y, **regressor)
    for nome_modelo, regressor in regressors.items()
}

df_resultados = organiza_resultados(resultados)
```

Depois:

```python
sns.boxplot(
    x="model",
    y="test_r2",
    data=df_resultados,
)
```

Se aparecer o erro:

```text
ValueError: Could not interpret value `model` for `x`
```

significa que o objeto passado em `data` nao possui a coluna `model`. Na pratica, isso costuma acontecer quando se usa `data=resultados` em vez de `data=df_resultados`.

Se aparecer:

```text
NameError: name 'df_resultados' is not defined
```

significa que a celula que cria `df_resultados = organiza_resultados(resultados)` nao foi executada com sucesso antes do grafico.

### Funcao de apoio

A funcao `organiza_resultados` transforma o dicionario do `cross_validate` em formato tabular, expandindo as metricas por fold e criando a coluna `model`.

Esse formato e adequado para:

- graficos com Seaborn;
- comparacao de distribuicao das metricas;
- analise de tempo de treino e score;
- comparacao visual entre modelos e preprocessamentos.

## Pontos Fundamentais para Revisao

### 1. EDA vem antes da modelagem

Antes de treinar um modelo, e necessario entender:

- tamanho da base;
- tipos de dados;
- valores ausentes;
- distribuicoes;
- correlacoes;
- possiveis outliers;
- comportamento do target.

Modelar sem EDA aumenta o risco de erro tecnico e interpretacao incorreta.

### 2. Treino e teste devem ser separados

Avaliacao em dados de treino nao mede generalizacao. Por isso, o modelo deve ser avaliado em dados que nao foram usados no `fit`.

### 3. Regressao linear e um bom baseline

A regressao linear e simples, interpretavel e rapida. Mesmo quando nao e o modelo final, ela serve como ponto de comparacao.

Um modelo mais complexo so vale a pena se superar esse baseline de forma consistente.

### 4. Metricas precisam ser adequadas ao problema

Para regressao, as metricas usadas ate agora foram adequadas:

- `MAE`: facil de interpretar;
- `MSE`: penaliza erros grandes;
- `RMSE`: fica na escala original do target;
- `R2`: mede proporcao de variancia explicada.

A ordem recomendada e sempre:

```python
metric(y_true, y_pred)
```

No projeto:

```python
metric(y_test, y_pred)
```

### 5. Coeficientes exigem cuidado

Coeficientes de regressao linear sao interpretaveis, mas:

- dependem da escala das variaveis;
- podem ser afetados por multicolinearidade;
- nao representam causalidade automaticamente;
- devem ser analisados junto com residuos e metricas.

Quando os dados passam por `StandardScaler`, a comparacao entre coeficientes fica mais justa.

### 6. Pipelines evitam vazamento de dados

Transformacoes como scaler, encoder e imputador devem ser ajustadas apenas nos dados de treino.

O `Pipeline` ajuda a garantir que:

- o `fit` seja feito no treino;
- o `transform` seja aplicado corretamente no treino e no teste;
- o fluxo seja reproduzivel.

### 7. Variaveis categoricas nao devem ser tratadas como numeros sem criterio

Uma categoria codificada como `1`, `2` ou `3` nao necessariamente possui ordem ou distancia numerica.

Quando nao ha ordem real, o tratamento correto geralmente e `OneHotEncoder`.

### 8. Curva de aprendizado ajuda a diagnosticar o modelo

A curva de aprendizado ajuda a responder:

- o modelo melhoraria com mais dados?
- o modelo esta sofrendo overfitting?
- o modelo esta sofrendo underfitting?
- o modelo e limitado demais para o problema?

### 9. Validacao cruzada reduz dependencia de uma unica divisao

Uma unica separacao treino/teste pode gerar uma estimativa instavel em bases pequenas. Com `KFold`, o modelo e avaliado em varias particoes, tornando a comparacao mais confiavel.

### 10. Resultados de validacao precisam ser organizados antes de plotar

O retorno de `cross_validate` e um dicionario de arrays. Para graficos comparativos, ele deve ser convertido para DataFrame.

No projeto, essa conversao e feita por:

```python
df_resultados = organiza_resultados(resultados)
```

O grafico deve usar `data=df_resultados`, nao `data=resultados`.

### 11. A versao do dataset precisa ser compativel com o preprocessador

Se o preprocessador usa `colesterol_hdl_cat`, a base correta e `DADOS_CATEGORIZADOS`.

Se a base carregada for `DADOS_TRATADOS`, essa coluna nao existe e o erro esperado e uma incompatibilidade entre as colunas esperadas pelo `ColumnTransformer` e as colunas disponiveis em `X`.

## Diagnostico Atual

O modelo atual e tecnicamente coerente como baseline, mas ainda limitado.

O desempenho de `R2` proximo de `0.453` indica que a regressao linear captura parte da relacao entre as variaveis e o target, mas deixa uma parcela relevante da variabilidade sem explicacao.

Isso pode acontecer por alguns motivos:

- relacoes nao lineares entre variaveis e target;
- interacoes entre variaveis nao modeladas;
- ruido natural nos dados clinicos;
- limite do proprio conjunto de variaveis disponiveis;
- multicolinearidade entre variaveis metabolicas.
