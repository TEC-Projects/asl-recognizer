import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sn


def plot_confusion_matrix(confusion_matrix):
    sn.heatmap(confusion_matrix, annot=True)
    plt.xlabel("Predicción")
    plt.ylabel("Objetivo")
    plt.show()


def plot_bar_chart(labels, data):
    plt.bar(labels, data, color='blue')
    plt.xlabel("Letras")
    plt.ylabel("Efectividad de predicción")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.show()
