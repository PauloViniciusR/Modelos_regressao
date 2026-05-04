import matplotlib.pyplot as plt

def plot_coefs(df_coeficientes):
    df_coeficientes.plot.barh()

    plt.axvline(x=0, color="0.5")
    plt.xlabel("Coeficiente")
    
    plt.gca().get_legend().remove()