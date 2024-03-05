from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

from analysis import createAnalysisValue, createStatesAnalysis


#from model import tflite_detect_image
#def runModel(imgpath):
#    modelpath = 'custom_model_lite/detect.tflite'
#    lblpath = 'custom_model_lite/labelmap.txt'  # Path to labelmap.txt file
#    img_processed, detection_results = tflite_detect_image(modelpath, imgpath, lblpath, min_conf=0.5, txt_only=False, savepath=None)
#    return img_processed, detection_results
def showImage(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')  # Remove os eixos para uma visualização mais limpa
    plt.show()

new_states = [1,1,1,1,1,1,1,1,1,1]
#print(estados)

# O append terá que ser feito no main, e então enviado para criar o estado de analises.
createStatesAnalysis(new_states)


#print(createAnalysisValue(estados))
#print(estados)
#showImage(img)
#img,detection_result = runModel("images/Screenshot_8.png")
#print(detection_result)
#estados = getState(detection_result)
#normal = 0
#atento = 1
#surpreso = 2
#dormindo = 3
#feliz = 4
#triste = 5
#raiva = 6
#entediado = 7
#detection_result tem estrutura: [{'estado': 'blob_atento', 'estado_numero': 1, 'prob': '0.9999999403953552'}]