from matplotlib import pyplot as plt

from model import tflite_detect_image
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

img,detection_result = runModel("images/Screenshot_2024-03-04_15-45-27.png")
print(detection_result)
showImage(img)