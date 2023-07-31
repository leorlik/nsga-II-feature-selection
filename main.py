from libs.metricas import *
from libs.resolucao import *
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sys

def export_specs():
    """
    Função que exporta as especificações das funções de métrica

    Parametros:
    -----------
    None

    Retorno:
    ------------
        Uma tupla de dicionários com as especificações das funções de métrica
    """

     ### Definição dos parâmetros da função de métrica
    ia_specs = {
        "func": intra_class_distance,
        "name": "intra_class_distance",
        "max": False,
        "abreviacao": "IA"
    }

    ie_specs = {
        "func": inter_class_distance,
        "name": "inter_class_distance",
        "max": True,
        "abreviacao": "IE"
    }

    ac_specs = {
        "func": attribute_class_correlation,
        "name": "attribute_class_correlation",
        "max": True,
        "abreviacao": "AC"
    }

    su_specs = {
        "func": fcbf,
        "name": "symmetrical_uncertainty",
        "max": False,
        "abreviacao": "SU"
    }

    ls_specs = {
        "func": laplacian_score,
        "name": "laplacian_score",
        "max": True,
        "abreviacao": "LS"
    }

    ### Criação da tupla para a iteração
    specs = (ia_specs, ie_specs, ac_specs, su_specs, ls_specs)

    return specs

def main(data_file):

    le = preprocessing.LabelEncoder()

    ### Transformações especificas do .data categórico
    if(".data" in data_file):
        dataset = np.loadtxt(data_file, delimiter = ',', dtype=str)
        X = []
        for i in range(1, dataset.shape[1]):
            X.append(le.fit_transform(dataset[:, [i]]))
        x = np.asarray(X)
        x = np.transpose(X)
        x = pd.DataFrame(x)

        y = dataset[:, [0]]
        y = le.fit_transform(y)
    ### Repartição específica do .csv
    elif (".csv" in data_file):
        dataset = pd.read_csv(data_file)
        ### Adquire as variáveis do dataset
        x = dataset.iloc[:, :-1]

        ### Adquire as classes do dataset
        y = dataset.iloc[:, -1]

        ### Transforma as classes em números
        y = le.fit_transform(y)
    else:
        print("Formato de arquivo nao reconhecido")
        sys.exit()

    specs = export_specs()

    scores = []
    
    for i in range(0, len(specs)):
        for j in range(i+1, len(specs)):

            file_name_1 = specs[i]["name"] + "_and_" + specs[j]["name"] + "_variables.dat"
            file_name_2 = specs[i]["name"] + "_and_" + specs[j]["name"] + "_scores.dat"

            features, fitness = resolve_problema(x , y, 
                                                    n_var = x.shape[1], func1 = specs[i], func2 = specs[j],
                                                    file_name_1 = file_name_1, file_name_2 = file_name_2, 
                                                    maximiza_1=specs[i]["max"], maximiza_2=specs[j]["max"])

            print("Caso de: " + specs[i]["name"] + " e " + specs[j]["name"])

            if(specs[i]["max"]):
                fitness[0] *= -1
            print("Valor de " + specs[i]["name"] + ": " +str(fitness[0]))

            if(specs[j]["max"]):
                fitness[1] *= -1
            print("Valor de " + specs[j]["name"] + ": "+ str(fitness[1]))

            acc, f1, recall, precision = avalia_solucao(x, y, features)

            scores.append((specs[i]["abreviacao"] + "+" + specs[j]["abreviacao"], acc, f1, recall, precision, fitness[0], fitness[1], np.sum(features)))

    ### Compara com todas as variáveis
    acc, f1, recall, precision = avalia_solucao(x, y, np.ones(x.shape[1]))
    scores.append(("Todas as variáveis", acc, f1, recall, precision, 0, 0, x.shape[1]))

    scores = pd.DataFrame(scores, columns=["Métricas", "Acurácia", "F1", "Recall", "Precisão", "Valor 1", "Valor 2", "Número de variáveis"])
    scores.to_csv("scores.csv", index=False)

if __name__ == "__main__":
    main(sys.argv[1])

