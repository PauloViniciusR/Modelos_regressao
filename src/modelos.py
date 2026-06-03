import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def treinar_e_validar_modelo_regressao(
    X,
    y,
    regressor,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
):

    validar_colunas_preprocessador(X, preprocessor)

    model = construir_pipeline_modelo(regressor, preprocessor, target_transformer)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

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

    return scores


def validar_colunas_preprocessador(X, preprocessor):
    if preprocessor is None or not hasattr(X, "columns"):
        return

    colunas_esperadas = obter_colunas_preprocessador(preprocessor)
    colunas_faltantes = sorted(set(colunas_esperadas) - set(X.columns))

    if colunas_faltantes:
        colunas_disponiveis = ", ".join(X.columns)
        colunas_faltantes_texto = ", ".join(colunas_faltantes)
        raise ValueError(
            "O preprocessador espera colunas que nao existem em X: "
            f"{colunas_faltantes_texto}. "
            f"Colunas disponiveis em X: {colunas_disponiveis}"
        )


def obter_colunas_preprocessador(preprocessor):
    colunas = []

    for _, _, transformer_columns in getattr(preprocessor, "transformers", []):
        if isinstance(transformer_columns, str):
            colunas.append(transformer_columns)
        elif isinstance(transformer_columns, (list, tuple)):
            colunas.extend(
                coluna for coluna in transformer_columns if isinstance(coluna, str)
            )

    return colunas


def construir_pipeline_modelo(regressor, preprocessor, target_transformer):
    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("reg", regressor)])
    else:
        pipeline = Pipeline([("reg", regressor)])

    if target_transformer is not None:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model = pipeline
    return model


def organiza_resultados(resultados):

    for chave, valor in resultados.items():
        resultados[chave]["time_seconds"] = (
            resultados[chave]["fit_time"] + resultados[chave]["score_time"]
        )

    df_resultados = (
        pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "model"})
    )

    df_resultados_expandido = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    df_resultados_expandido.columns = df_resultados_expandido.columns.str.strip()

    colunas_numericas = df_resultados_expandido.columns.drop("model")
    df_resultados_expandido[colunas_numericas] = df_resultados_expandido[
        colunas_numericas
    ].apply(pd.to_numeric)

    return df_resultados_expandido
