import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


def construir_pipeline_modelo(regressor, preprocessor=None, target_transformer=None):
    """Monta o estimador final com preprocessamento e transformacao do target."""
    if preprocessor is not None:
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("reg", regressor),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("reg", regressor),
            ]
        )

    if target_transformer is not None:
        modelo = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=target_transformer,
        )
    else:
        modelo = pipeline

    return modelo


def treinar_e_validar_modelo_regressao(
    X,
    y,
    regressor,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
):
    modelo = construir_pipeline_modelo(
        regressor=regressor,
        preprocessor=preprocessor,
        target_transformer=target_transformer,
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_validate(
        modelo,
        X,
        y,
        cv=kf,
        scoring=[
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
    )

    return scores


def organiza_resultados(scores, nome_modelo=None):
    """Organiza os resultados da validacao cruzada em uma tabela resumida."""
    if isinstance(scores, str) and nome_modelo is not None:
        scores, nome_modelo = nome_modelo, scores

    resultados = pd.DataFrame(scores).copy()

    colunas_erro = [
        coluna
        for coluna in resultados.columns
        if coluna.startswith("test_neg_")
    ]

    # O scikit-learn retorna erros como valores negativos para manter a logica
    # de que quanto maior o score, melhor. Para leitura, voltamos ao valor positivo.
    resultados[colunas_erro] = resultados[colunas_erro].abs()
    resultados = resultados.rename(
        columns={
            coluna: coluna.replace("test_neg_", "test_")
            for coluna in colunas_erro
        }
    )

    colunas_metricas = [
        coluna
        for coluna in resultados.columns
        if coluna.startswith("test_")
    ]

    resultados_resumidos = (
        resultados[colunas_metricas]
        .agg(["mean", "std"])
        .T
        .reset_index()
        .rename(
            columns={
                "index": "metrica",
                "mean": "media",
                "std": "desvio_padrao",
            }
        )
    )

    if nome_modelo is not None:
        resultados_resumidos.insert(0, "modelo", nome_modelo)

    return resultados_resumidos
