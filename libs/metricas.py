import numpy as np
import pandas as pd

def shell_func(x, x_index, func, y):
    """
    Função que evalua uma função de métrica para um subconjunto de variáveis de um dataframe

    Parametros:
    -----------
        x: dataframe alvo
        x_index: vetor de booleanos que indica quais variáveis do dataframe serão consideradas
        func: função de métrica
        y: classes

    Retorno:
    ------------
        O valor da métrica da função func para o subconjunto de variáveis indicado por x_index
    """
    
    ### Coloca em vars o índice das variáveis que serão consideradas
    vars = []
    for i, var in enumerate(x_index):
        if(var):
            vars.append(i)

    ### Adquire o subconjunto de variáveis do dataframe que serão avaliadas pela função func
    x_evaluate = x.values[:, vars]

    ### Se não há variáveis, retorna 0
    if(x_evaluate.shape[1] == 0):
        return 9999

    return func(x_evaluate, y)

def intra_class_distance(x_local, y_local):
    """
    Função que calcula a distância intra-classe de um conjunto de dados

    Parametros:
    ------------
        x_local: conjunto de dados
        y_local: classes

    Retorno:
    ------------
        A distância intra-classe do conjunto de dados x_local
    """
    
    ### Adquire o número de classes
    classes = np.unique(y_local)

    #Cria uma lista indexada pelas classes
    x_classes = []
    for classe in classes:
        x_classes.append([])

    ### Coloca cada elemento de x_local na classe equivalente na lista x_classes
    for i in range(0, len(x_local)):
        j = 0
        while(y_local[i] != j):
            j+=1
        x_classes[j].append(x_local[i])

    ### Soma as distâncias intra-classe de todas as classes
    distance = 0
    for x_target in x_classes:
        avg_point = np.average(np.asarray(x_target))
        for x_ind in x_classes:
            distance += np.linalg.norm(x_ind - avg_point)

    ### Retorna a média das distâncias intra-classe de todas as classes
    return distance/x_local.shape[1]

def inter_class_distance(x_local, y_local):
    """
    Função que calcula a distância inter-classe de um conjunto de dados

    Parametros:
    ------------
        x_local: conjunto de dados
        y_local: classes

    Retorno:
    ------------
        A distância inter-classe do conjunto de dados x_local
    """
    
    ### Adquire o número de classes
    classes = np.unique(y_local)

    #Cria uma lista indexada pelas classes
    x_classes = []
    for classe in classes:
        x_classes.append([])

    ### Coloca cada elemento de x_local na classe equivalente na lista x_classes
    for i in range(0, len(x_local)):
        j = 0
        while(y_local[i] != j):
            j+=1
        x_classes[j].append(x_local[i])

    ### Soma as distâncias inter-classe de todas as classes
    distance = 0
    for i in range(0, len(x_classes)):
        for j in range(i+1, len(x_classes)):
            distance += np.linalg.norm(np.average(np.asarray(x_classes[i])) - np.average(np.asarray(x_classes[j])))

    ### Retorna a média das distâncias inter-classe de todas as classes
    return distance/x_local.shape[1]

def class_entropy(x_local, y_local):

    """
    Função que calcula a entropia de um conjunto de dados

    Parametros:
    ------------
        x_local: conjunto de dados
        y_local: classes

    Retorno:
    ------------
        A entropia do conjunto de dados x_local
    """
    
    ### Adquire o número de classes
    classes = np.unique(y_local)

    #Cria uma lista indexada pelas classes
    x_classes = []
    for classe in classes:
        x_classes.append([])

    ### Coloca cada elemento de x_local na classe equivalente na lista x_classes
    for i in range(0, len(x_local)):
        j = 0
        while(y_local[i] != j):
            j+=1
        x_classes[j].append(x_local[i])

    ### Calcula a entropia de cada classe
    entropies = []
    for x_target in x_classes:
        entropy = 0
        for x_ind in x_target:
            entropy += x_ind*np.log(x_ind)
        entropies.append(entropy)

    ### Retorna a média das entropias de todas as classes
    return np.average(np.asarray(entropies))

from sklearn.feature_selection import mutual_info_classif as MIC

def attribute_class_correlation(x_local, y_local):
    """
    Função que calcula a correlação de atributos de classe entre as variáveis de um conjunto de dados e suas classes

    Parametros:
    ------------
        x_local: conjunto de dados
        y_local: classes

    Retorno:
    ------------
        A correlação de atributos de classe do conjunto de dados x_local
    """
    return (MIC(x_local, y_local).sum())/x_local.shape[0]

from skfeature.function.information_theoretical_based import FCBF
from skfeature.utility.mutual_information import su_calculation
def fcbf(X, y, **kwargs):
    """

    Esta função possuia problemas em sua implementação e fiz alguns ajustes na tal para que ela funcionasse corretamente
    This function implements Fast Correlation Based Filter algorithm
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        delta: {float}
            delta is a threshold parameter, the default value of delta is 0
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    SU: {numpy array}, shape (n_features,)
        symmetrical uncertainty of selected features
    Reference
    ---------
        Yu, Lei and Liu, Huan. "Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution." ICML 2003.
    """

    n_samples, n_features = X.shape
    if 'delta' in kwargs.keys():
        delta = kwargs['delta']
    else:
        # the default value of delta is 0
        delta = 0

    # t1[:,0] stores index of features, t1[:,1] stores symmetrical uncertainty of features
    t1 = np.zeros((n_features, 2), dtype='object')
    for i in range(n_features):
        f = X[:, i]
        t1[i, 0] = i
        t1[i, 1] = su_calculation(f, y)
    s_list = t1[t1[:, 1] > delta, :]
    # index of selected features, initialized to be empty
    F = []
    # Symmetrical uncertainty of selected features
    SU = []
    while len(s_list) != 0:
        # select the largest su inside s_list
        idx = np.argmax(s_list[:, 1])
        # record the index of the feature with the largest su
        fp = X[:, s_list[idx, 0]]
        np.delete(s_list, idx, 0)
        F.append(s_list[idx, 0])
        SU.append(s_list[idx, 1])
        for i in s_list[:, 0]:
            fi = X[:, i]
            if su_calculation(fp, fi) >= t1[i, 1]:
                # construct the mask for feature whose su is larger than su(fp,y)
                idx = s_list[:, 0] != i
                idx = np.array([idx, idx])
                idx = np.transpose(idx)
                # delete the feature by using the mask
                s_list = s_list[idx]
                length = len(s_list)//2
                s_list = s_list.reshape((length, 2))
    #return np.array(F, dtype=int), np.array(SU)
    su = np.array(SU)
    return (np.sum(su))/su.shape[0]

from skfeature.function.similarity_based import lap_score
from skfeature.utility.construct_W import construct_W

def laplacian_score(x_local, y_local, kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}):
    """
    Função que calcula a pontuação laplaciana de um conjunto de dados

    Parametros:
    ------------
        x: conjunto de dados completo
        x_local: subconjunto de dados
        y_local: classes
        kwargs_W: parametros para a construção da matriz de pesos

    Retorno:
    ------------
        A pontuação laplaciana do conjunto de dados x_local
    """
    W = construct_W(x_local, **kwargs_W)
    score = lap_score.lap_score(x_local, W=W)
    return (np.sum(score)/len(score))