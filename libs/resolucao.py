from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from libs.metricas import shell_func
import numpy as np
import os

def manhattan_distance( xy1, xy2 ):
    """
    Função que calcula a distância de manhattan entre dois pontos

    Parametros:
    ------------
        xy1: coordenadas do ponto 1
        xy2: coordenadas do ponto 2

    Retorno:
    ------------
        A distância de manhattan entre os dois pontos
    """
    return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )

def get_best_index(F, max_0, max_1):
    """
    Função que retorna o índice do ponto de F mais próximo do ponto (max_0, max_1)

    Parametros:
    ------------
        F: matriz de pontos
        max_0: coordenada x do ponto
        max_1: coordenada y do ponto

    Retorno:
    ------------
        O índice do ponto de F mais próximo do ponto (max_0, max_1)
    """
    point = np.asarray([max_0, max_1])
    dist = manhattan_distance(point, F[0])
    index = 0
    for i in range(1, F.shape[0]):
        ### Se a distancia é 9999, não considera o ponto
        if(F[i][0] == 9999 or F[i][1] == 9999):
            continue
        dist_atual = manhattan_distance(point, F[i])
        if (dist_atual < dist):
            dist = dist_atual
            index = i
    
    return(index)

def resolve_problema(x, y, func1, func2, n_var, file_name_1 = "x.dat", file_name_2 = "f.dat", save_files = True,
                        maximiza_1 = False, maximiza_2 = False, pop_size = 50, n_gen = 50):
    """
    Função que resolve um problema de otimização multi-objetivo utilizando o algoritmo NSGA-II e retorna o melhor indivíduo e seu fitness

    Parametros:
    ------------
        x: variaveis do problema
        y: classes do problema
        func1: função objetivo 1
        func2: função objetivo 2
        n_var: número de variáveis
        file_name_1: nome do arquivo que será salvo o genótipo do melhor indivíduo
        file_name_2: nome do arquivo que será salvo o fitness do melhor indivíduo
        save_files: se True, salva os arquivos
        maximiza_1: se True, maximiza a função objetivo 1
        maximiza_2: se True, maximiza a função objetivo 2
        pop_size: tamanho da população
        n_gen: número de gerações

    Retorno:
    ------------
        O melhor indivíduo e seu fitness
    """

    RESULTS_DIR = "resultados"

    ### Cria o algoritmo NSGA-II
    algorithm = NSGA2(pop_size=pop_size,
                sampling=BinaryRandomSampling(),
                crossover=TwoPointCrossover(prob=0.8),
                mutation=BitflipMutation(),
                eliminate_duplicates=True)

    ### Utiliza o fator para multiplicar por -1 as funções objetivos que devem ser maximizada, pois o solver
    ### tenta minimizar todas as funções objetivos
    fator_1 = 1
    fator_2 = 1
    if(maximiza_1):
        fator_1 = -1
    if(maximiza_2):
        fator_2 = -1

    ### Define os objetivos do problema multi-objetivo
    objetivo = [
        lambda a: fator_1 * shell_func(x, a, func1["func"], y),
        lambda a: fator_2 * shell_func(x, a, func2["func"], y)
    ]

    ### Define a restrição que a soma dos pesos deve ser maior que 0 (De uma forma que utilize o padrão do framework, ou seja, que seja uma inequalidade <= 0)
    constr_ieq = [
        lambda a: (-1 *np.sum(a)) +1
    ]   
    
    ### Define a função de parada
    termination = get_termination("n_gen", n_gen)

    ### Cria o problema multi-objetivo
    problem = FunctionalProblem(n_var,
                                objetivo,
                                constr_ieq=constr_ieq,
                                xl= np.array([0] * n_var),
                                xu= np.array([1] * n_var))

    ### Resolve o problema
    res = minimize(problem, algorithm, termination, seed = 98, verbose = False)

    ### Pega os melhores indivíduos e seus fitness
    x_res = res.X.astype(int)
    f_res = res.F

    ### Pega os melhores fitness
    better_f1 = min(f_res[:,[0]])
    better_f2 = min(f_res[:,[1]])

    ### Utiliza a função get_best_index para decidir entre os melhores indivíduos qual é o melhor dentro do critério estabelecido
    index = get_best_index(f_res, better_f1, better_f2)

    ### Adquirie o indivíduo que melhor se encaixa nos padrões estabelecidos
    best_x = x_res[index]
    best_f = f_res[index]

    ### Salva os arquivos caso save_files seja True
    if(save_files):
        ### Cria a pasta de resultados caso não exista
        if(not os.path.isdir(RESULTS_DIR)):
            os.mkdir(RESULTS_DIR)

        ### Salva os arquivos
        best_x.tofile(os.path.join(RESULTS_DIR, file_name_1))
        best_f.tofile(os.path.join(RESULTS_DIR, file_name_2))

    ### Retorna o melhor indivíduo e seu fitness
    return best_x, best_f

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import FitFailedWarning   

def avalia_solucao(x, y, features_x, params = {'penalty': ['l2', 'elasticnet'], 'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs', 'saga']}):
    """
    Função que avalia a performance do algoritmo de regressão logística em um subconjunto de variáveis do dataset x

    Parametros:
    ------------
        x: dataset de variáveis
        y: classes do problema
        features_x: vetor que indica os pesos de cada variável (0 ou 1)
        params: parâmetros do GridSearchCV

    Retorno:
    ------------
        Acurácia, F1, Recall e Precisão do melhor modelo construído em cima daquela solução
    """

    ### Função interna que adquire os indexes do subconjunto de variáveis selecionado
    def get_all_indexes(solution):
        s = []
        for i in range(0, solution.shape[0]):
            if solution[i] == 1:
                s.append(i)
        return s

    acc_soma = 0.0
    f1_soma = 0.0
    recall_soma = 0.0
    precision_soma = 0.0

    ### Adquire as variáveis que foram selecionadas
    xv = x.values
    indexes = get_all_indexes(features_x)
    xv = xv[:, indexes]

    kf = KFold(shuffle=True,random_state=66,n_splits=10)

    ### Faz a validação cruzada
    for Itr, Ite in kf.split(xv):
        gs = GridSearchCV(LogisticRegression(), params, cv = 5,scoring='accuracy', verbose=False)

        ### Separa os dados de treino e teste 
        X_tr, X_te, y_tr, y_te = xv[Itr,:], xv[Ite,:], y[Itr], y[Ite]

        ### Silencia os warnings do GridSearchCV
        with ignore_warnings(category=(FitFailedWarning, UserWarning)):
            gs.fit(X_tr, y_tr)

        ### Prediz e calcula as métricas de interesse
        y_pred = gs.predict(X_te)
        acc_soma += accuracy_score(y_te,y_pred)
        f1_soma += f1_score(y_te,y_pred)
        recall_soma += recall_score(y_te,y_pred)
        precision_soma += precision_score(y_te,y_pred)

    ### Calcula as médias das métricas
    acc_media =  acc_soma / kf.n_splits
    f1_media =  f1_soma / kf.n_splits
    recall_media =  recall_soma / kf.n_splits
    precision_media =  precision_soma / kf.n_splits

    return acc_media, f1_media, recall_media, precision_media


