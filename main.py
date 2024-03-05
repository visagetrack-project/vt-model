from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from analysis import createStatesAnalysis, getState

from analysis import createAnalysisValue, createStatesAnalysis
from model import tflite_detect_image


#from model import tflite_detect_image
def runModel(imgpath):
    modelpath = 'custom_model_lite/detect.tflite'
    lblpath = 'custom_model_lite/labelmap.txt'  # Path to labelmap.txt file
    img_processed, detection_results = tflite_detect_image(modelpath, imgpath, lblpath, min_conf=0.5, txt_only=False, savepath=None)
    return img_processed, detection_results
def showImage(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')  # Remove os eixos para uma visualização mais limpa
    plt.show()

def processImages(folder_path, max_images=50):
    estados_accum = []  # Lista para acumular estados de todas as imagens

    for i in range(1, max_images + 1):
        img_path = os.path.join(folder_path, f'Screenshot_{i}.png')

        # Verifica se a imagem existe
        if not os.path.exists(img_path):
            print(f"Imagem {img_path} não encontrada.")
            break  # Sai do loop se a imagem não existir

        # Processa a imagem com o modelo
        img_processed, detection_results = runModel(img_path)

        # Converte os resultados de detecção em estados
        estados = getState(detection_results)
        print(estados)
        # Exemplo de estados mockados, substituir pela linha acima quando a implementação estiver pronta

        estados_accum.append(estados)  # Acumula estados


    # Após processar todas as imagens, passa os estados acumulados para análise
    print(estados_accum)
    createStatesAnalysis(estados_accum)


# Caminho para a pasta que contém as imagens
images_folder_path = 'images'
# Executa o processamento das imagens
processImages(images_folder_path)
