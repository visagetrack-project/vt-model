import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Calcula o valor de engajamento com base nos estados
def createAnalysisValue(estados, blobQuantity=10):
    total_max_value = 50
    total_min_value = 0
    total_value = 0
    for estado in estados:
        if estado == 0:  # normal
            total_value += (5 / blobQuantity)
        elif estado == 1:  # atento
            total_value += (50 / blobQuantity)
        elif estado == 2:  # surpreso
            total_value += (80 / blobQuantity)
        elif estado == 3:  # dormindo
            total_value -= (80 / blobQuantity)
        elif estado == 4:  # feliz
            total_value += (30 / blobQuantity)
        elif estado == 5:  # triste
            total_value -= (30 / blobQuantity)
        elif estado == 6:  # raiva
            total_value -= (50 / blobQuantity)
        elif estado == 7:  # entediado
            total_value -= (25 / blobQuantity)
        else:
            total_value -= (5 / blobQuantity)
    total_value = max(min(total_value, total_max_value), total_min_value)
    return total_value

# Realiza a regressão linear nos valores de engajamento
def realizar_regressao_linear(X, y):
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo

# Mostra a regressão e os dados originais
def criarRegressao(momentos, valores_engajamento):
    modelo_regressao = realizar_regressao_linear(momentos, valores_engajamento)
    valores_previstos = modelo_regressao.predict(np.array(momentos).reshape(-1, 1))
    plt.figure(figsize=(10, 6))
    plt.scatter(momentos, valores_engajamento, color='blue', label='Dados Originais')
    plt.plot(momentos, valores_previstos, color='red', label='Linha de Regressão')
    plt.title('Engajamento ao Longo do Tempo com Regressão Linear')
    plt.xlabel('Momento')
    plt.ylabel('Engajamento')
    plt.legend()
    plt.grid(True)
    plt.xticks(momentos)
    plt.show()
# Função para adicionar novo conjunto de estados e realizar a análise
def createStatesAnalysis(estados_conjunto):
    valores_engajamento = [createAnalysisValue(estados) for estados in estados_conjunto]
    momentos = list(range(1, len(valores_engajamento) + 1))
    criarRegressao(momentos, valores_engajamento)

def getState(detection_result):
    estados = []  # Inicializa a lista para armazenar os valores de estado_numero
    for item in detection_result:  # Itera sobre cada item em detection_result
        estado_numero = item['estado_numero']  # Extrai o valor de estado_numero
        estados.append(estado_numero)  # Adiciona o valor à lista estados
    return estados

