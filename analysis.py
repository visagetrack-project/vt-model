import glob
import os
import time

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Calcula o valor de engajamento com base nos estados
def createAnalysisValue(estados, blobQuantity=10):
    total_max_value = 50
    total_min_value = -50
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

def criarRegressao(momentos, valores_engajamento, save_dir=r"C:\Users\jader\Desktop\estudos\visage-track\vt-api\packages\graph"):
    files = glob.glob(os.path.join(save_dir, '*.png'))
    for f in files:
        os.remove(f)
    modelo_regressao = realizar_regressao_linear(momentos, valores_engajamento)
    valores_previstos = modelo_regressao.predict(np.array(momentos).reshape(-1, 1))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_plots = len(momentos)
    for i in range(1, total_plots + 1):
        if i % 10 == 0 or i == total_plots:  # Salva a cada 10 plotagens ou na última
            plt.figure(figsize=(10, 6))
            plt.scatter(momentos[:i], valores_engajamento[:i], color='blue', label='Dados Originais')  # Pontos de dados originais até o momento i
            plt.plot(momentos[:i], valores_engajamento[:i], 'k-', label='Linha Conectando Pontos')  # Linha preta conectando os pontos
            plt.plot(momentos[:i], valores_previstos[:i], color='red', label='Linha de Regressão')  # Linha de regressão até o momento i
            plt.title('Engajamento ao Longo do Tempo com Regressão Linear')
            plt.xlabel('Momento')
            plt.ylabel('Engajamento')
            plt.legend()
            plt.grid(True)
            plt.xticks(momentos[:i])

            # Salva a figura atual
            figure_path = os.path.join(save_dir, f"regressao_{i}.png")
            plt.savefig(figure_path)
            plt.close()  # Fecha a figura atual para que um novo plot possa ser iniciado

    # Exibe a última figura após salvar todas as imagens intermediárias
    plt.figure(figsize=(10, 6))
    plt.scatter(momentos, valores_engajamento, color='blue', label='Dados Originais')
    plt.plot(momentos, valores_engajamento, 'k-', label='Linha Conectando Pontos')  # Linha preta conectando os pontos em todos os dados
    plt.plot(momentos, valores_previstos, color='red', label='Linha de Regressão')
    plt.title('Engajamento ao Longo do Tempo com Regressão Linear')
    plt.xlabel('Momento')
    plt.ylabel('Engajamento')
    plt.legend()
    plt.grid(True)
    plt.xticks(momentos)
    print("Abra A API EM GO!")
    time.sleep(10)
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

