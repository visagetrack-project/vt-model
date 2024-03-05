# Convert exported graph file into TFLite model file
import csv

import tensorflow as tf
import sys
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from tensorflow.lite.python.interpreter import Interpreter
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import os

converter = tf.lite.TFLiteConverter.from_saved_model('custom_model_lite/saved_model')
tflite_model = converter.convert()

with open('custom_model_lite/detect.tflite', 'wb') as f:
  f.write(tflite_model)

  # Script to run custom TFLite model on test images to detect objects
  # Source: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_image.py


# Function to generate XML file for detected objects
def generate_xml(imgpath, labels, boxes, scores, classes, min_conf, imW, imH, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)  # Cria o diretório se não existir

    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = os.path.basename(imgpath)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(imW)
    ET.SubElement(size, "height").text = str(imH)
    ET.SubElement(size, "depth").text = "3"

    for i, score in enumerate(scores):
        if score > min_conf:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = labels[int(classes[i])]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = str(0)
            ET.SubElement(obj, "difficult").text = str(0)
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(boxes[i][1] * imW))
            ET.SubElement(bndbox, "ymin").text = str(int(boxes[i][0] * imH))
            ET.SubElement(bndbox, "xmax").text = str(int(boxes[i][3] * imW))
            ET.SubElement(bndbox, "ymax").text = str(int(boxes[i][2] * imH))

    xml_filename = os.path.splitext(os.path.basename(imgpath))[0] + '.xml'
    xml_filepath = os.path.join(savepath, xml_filename)
    tree = ET.ElementTree(root)
    tree.write(xml_filepath)

### Define function for inferencing with TFLite model and displaying results
def tflite_detect_image(modelpath, imgpath, lblpath, min_conf=0.5, txt_only=False, savepath=None):

    # Load the label map
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load TFLite model and allocate tensors.
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]['shape'][1:3]

    float_input = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    # Load image
    image = cv2.imread(imgpath)
    if image is None:
        print(f"Erro ao carregar a imagem: {imgpath}. Verifique o caminho do arquivo e as permissões.")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e., if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

    print(boxes)
    for score in scores:
        print(f'{score:.16f}%')
    print(classes)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(max(1, (xmin * imW)))
            xmax = int(min(imW, (xmax * imW)))
            ymin = int(max(1, (ymin * imH)))
            ymax = int(min(imH, (ymax * imH)))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])]

            label = '{}: {:.16f}%'.format(object_name, scores[i] * 100)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # Lista para armazenar os resultados das detecções
    detection_results = []
    csv_filename = "detection_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escreve o cabeçalho do CSV
        writer.writerow(['estado', 'estado_numero', 'prob'])

        # Escreve as linhas com os resultados da detecção
        for i in range(len(scores)):
                estado = labels[int(classes[i])]
                estado_numero = int(classes[i])
                prob = scores[i]
                writer.writerow([estado, estado_numero, f'{prob:.16f}'])
                detection = {
                    'estado': labels[int(classes[i])],  # O rótulo da classe detectada
                    'estado_numero': int(classes[i]),  # O índice numérico da classe
                    'prob': f'{scores[i]:.16f}'  # A probabilidade da detecção, formatada
                }
                detection_results.append(detection)

    # Convert the image back to BGR for OpenCV compatibility
    image_processed = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Optionally save detection results in a .txt file
    if txt_only and savepath is not None:
        image_fn = os.path.basename(imgpath)
        base_fn, _ = os.path.splitext(image_fn)
        txt_result_fn = base_fn + '.txt'
        txt_savepath = os.path.join(savepath, txt_result_fn)

        with open(txt_savepath, 'w') as f:
            for i in range(len(scores)):
                if scores[i] > min_conf and scores[i] <= 1.0:
                    f.write('%s %.16f %d %d %d %d\n' % (labels[int(classes[i])], scores[i], xmin, ymin, xmax, ymax))
    image_processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # No final da função tflite_detect_image, adicione a chamada para generate_xml
    generate_xml(imgpath, labels, boxes, scores, classes, min_conf, imW, imH,savepath='test.xml')

    return image_processed,detection_results

