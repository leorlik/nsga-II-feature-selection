import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_bar_chart(df, metric_column, accuracy_column, title):

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the bar chart
    ax.bar(df[metric_column], df[accuracy_column])
    ax.set_xlabel(metric_column)
    ax.set_ylabel(accuracy_column)
    ax.set_title(title)

    for bars in ax.containers:
        ax.bar_label(bars, fontsize=6)

    # Show the chart
    # plt.show()

    # Save the chart
    fig.savefig("figuras/" + title + ".png", dpi=300, bbox_inches='tight')

def bar_chart_metrics(df, metric_column, data_name, second_column):

    df["Porcentagem de variáveis utilizadas"] = df["Número de variáveis"]/df["Número de variáveis"].max()

    metrics = ("Acurácia", "F1", "Recall", "Precisão", "Porcentagem de variáveis utilizadas")
    categories = {
        "Todas as variáveis": tuple(df.loc[df[metric_column] == "Todas as variáveis"].drop(columns=["Valor 1", "Valor 2", "Métricas", "Número de variáveis"]).values[0]),
        second_column: tuple(df.loc[df[metric_column] == second_column].drop(columns=["Valor 1", "Valor 2", "Métricas", "Número de variáveis"]).values[0])
    }

    x = np.arange(len(metrics))
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(figsize=(12, 7))

    for attribute, measurement in categories.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fontsize=6)
        multiplier += 1

    ax.set_ylabel("Valores")
    ax.set_title('Comparação de métricas para o dataset de {}'.format(data_name))
    ax.set_xticks(x + width, metrics)
    ax.legend(loc='upper left', ncols=2)

    # plt.show()
    
    # Save the chart
    fig.savefig("figuras/" + "Comparação de métricas para o dataset de {}".format(data_name) + ".png", dpi=300, bbox_inches='tight')





def main():

    df_sonar = pd.read_csv("scores_sonar.csv")
    df_mushroom = pd.read_csv("scores_mushroom.csv")

    metric_column = "Métricas"
    accuracy_column = "Acurácia"
    create_bar_chart(df_sonar, metric_column, accuracy_column, title = "Acurácia para dataset de Sonar")
    create_bar_chart(df_mushroom, metric_column, accuracy_column, title = "Acurácia para dataset de Cogumelos")

    # Selecting only "Todas" and IE+AC lines
    df_sonar = df_sonar.loc[df_sonar[metric_column].isin(["Todas as variáveis", "IE+AC"])]

    # Selecting only "Todas" and IA+AC lines
    df_mushroom = df_mushroom.loc[df_mushroom[metric_column].isin(["Todas as variáveis", "IA+AC"])]

    bar_chart_metrics(df_sonar, metric_column, "Sonar", "IE+AC")
    bar_chart_metrics(df_mushroom, metric_column, "Cogumelos", "IA+AC")

if __name__ == "__main__":
    main()
