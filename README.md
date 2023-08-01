<strong>Seleção de Variáveis a partir de algoritmo genético multi-objetivo</strong>
========================================

## 1. Apresentação

Este projeto surgiu à partir de um projeto de graduação na matéria "Tópicos em Computação Bioinspirada", ministrada pelo professor Eduardo Spinosa na Universidade Federal do Paraná, no início de 2023, e majoritariamente inspirado pelos artigos **Venkatadri.M and Rao.K, S. (2010). A multiobjective genetic algorithm for feature selection in data mining** e **Yu, L. and Liu, H. (2003). Feature selection for high-dimensional data: A fast correlation-based filter solution.**

O objetivo deste trabalho é avaliar o impacto da seleção de variáveis em uma abordagem estatística independente de classificador, juntamente com um olhar minucioso sobre cada uma das métricas escolhidas e como minimizar as variáveis impacta o modelo geral. Ao contrário do que pode-se pensar, a análise aqui não tem o objetivo de maximizar a acurácia, a precisão ou escolher o melhor modelo, mas preocupa-se muito mais em analisar o processo e se a base teórica é comprovada ou não através de experimentos empíricos.

<br>

## 2. Estrutura do projeto:

- **datasets**: contém os arquivos de dados avaliados nos experimentos.

- **libs**: contém arquivos .py que possuem funçõesu tilizadas no programa principal.

- **enviroment.yml**: arquivo que descreve a versão do python e das bibliotecas utilizadas. Tal arquivo foi gerado pelo anaconda. Para estabelecer o ambiente, rode:

```bash
conda env create -f environment.yml
```

- **main.py**: arquivo principal que realiza os experimentos.

## 3. Algoritmos genéticos:

Algoritmos genéticos utilizam de populações para desenvolver, durante uma iteração, uma evolução de sua população atual. Ao contrário dos populares algoritmos de otimização que desenvolvem informações de gradiente no processo de busca a cada solução testada, o algoritmo genético testa várias soluções em um processo de busca direto, utilizando operadores estocásticos, e não os determinísticos tradicionais da otimização.

O funcionamento básico de um algoritmo genético consiste em iniciar uma população aleatória dentro dos limites predefinidos, evaluar a qualidade da solução através da função de avaliação de performance escolhida, e, com base nessa pontuação, atribuir uma probabilidade de escolha dessas soluções para o cruzamento. Importante ressaltar que não necessariamente as melhores soluções serão escolhidas para o cruzamento, embora isso possa ser garantido de alguma forma utilizando estratégias de elitismo (algumas radicalizações de tal estratégia fazem até mesmo as melhores soluções progredirem para a próxima população), mas tem mais chance, balanceando assim a exploração local e a global

## 4. Algoritmos genéticos multi-objetivo:

Uma aproximação multi-objetivo de algoritmos genéticos envolve a inclusão de mais de uma função objetivo a ser maximizada ou minimizada. Vale ressaltar que isso não limita as funções objetivo de um certo problema todas serem de maximização ou minimização - neste próprio trabalho, conforme explorado mais pra frente, há momentos em que funções para serem maximizadas e minimizadas são usadas. 

No caso da otimização multi-objetivo as funções objetivos constroem um espaço multi-dimensional chamado espaço objetivo, onde há uma projeção de cada solução neste espaço objetivo.

## 5. Fronteira de Pareto:

Num problema de otimização multi-objetivo, as soluções ótimas (afinal, um problema de otimização pode ter nenhuma, uma, várias ou infinitas soluções), são aquelas que são as primeiras numa ordenação parcial. Para que se entenda isso, deve-se utilizar o noção de dominância entre soluções. A definição de dominância se dá que uma solução $x\_{1}$ domina $x\_{2}$ se $x\_{1}$ não é pior que $x\_{2}$ em todos os objetivos e é melhor que $x\_{2}$ em pelo menos um objetivo. Quando não se há soluções que dominam x, x é uma solução não-dominada. O conjunto de soluções não dominadas são chamados de fronteira não-dominada. As projeções pertencentes a fronteira não-dominada são boas em um objetivo, mas podem ser pior que outras em determinados objetivos: essa troca entre os pontos não-dominados fazem os algoritmos se interessarem em encontrar uma variedade da soluções antes de tomar uma decisão final.

Os pontos não dominados por nenhuma solução no espaço objetivo são os pontos ótimos de pareto, enquanto o vetor de variáveis que projetam tais pontos são chamadas de soluções ótimas de Pareto. O objetivo dos algoritmos multi-objetivos, são, portanto, encontrar o máximo de pontos ótimos de pareto, para que algum algoritmo de decisão escolha a solução com o melhor balanceamento entre os objetivos.

## 6. NSGA-II:

m Abril de 2002 foi proposto a evolução do nondominated sorting genetic algorithm (algoritmo genético de ordenção não-dominada), o NSGA-II\cite{Deb:02}, que hoje em dia conta com 35800 citações(Connected Papers, acesso em 07/02/2023). Tendo populaçoes de tamanho N, o NSGA-II recombina tal população numa população de tamanho 2N e a ordena baseado no ranking da fronteira de pareto, priorizando soluções da fronteira classe 1 e completando com as fronteiras subsequentes, e o desempate de soluções na mesma classe de fronteira se dá por um operador chamado _crowd distance_, que consegue cobrir a maior quantia do espaço amostral daquela fronteira de Pareto.

## 7. Funções objetivos:

## 8. Parâmetros do NSGA-II:

## 9. Datasets:

## 10. Resultados:

## 11. Conclusão e experimentos futuros:
