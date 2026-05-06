import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import PredictionErrorDisplay

def plot_coefs(df_coeficientes):
    df_coeficientes.plot.barh()

    plt.axvline(x=0, color="0.5")
    plt.xlabel("Coeficiente")
    
    plt.gca().get_legend().remove()


def plot_residuos(y_true, y_pred):
    residuos = y_true - y_pred

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    h = sns.histplot(residuos, kde=True, ax=axs[0])

    error_display_01 = PredictionErrorDisplay.from_predictions(
    y_true=y_true,
    y_pred=y_pred,
    ax=axs[1]
)
    
    error_display_02 = PredictionErrorDisplay.from_predictions(
    y_true=y_true,
    y_pred=y_pred,
    kind='actual_vs_predicted',
    ax=axs[2]
)
    
    plt.tight_layout()

    plt.show()