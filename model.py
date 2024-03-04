# Convert exported graph file into TFLite model file
import tensorflow as tf
import sys

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
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

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
                    f.write('%s %.4f %d %d %d %d\n' % (labels[int(classes[i])], scores[i], xmin, ymin, xmax, ymax))
    image_processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_processed


# Set up variables for running user's model
imgpath='images/Screenshot_2024-03-04_15-45-31.png'   # Path to test images folder
modelpath='custom_model_lite/detect.tflite'   # Path to .tflite model file
lblpath='custom_model_lite/labelmap.txt'   # Path to labelmap.txt file
min_conf_threshold=0.5   # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
images_to_test = 6   # Number of images to run detection on

# Run inferencing function!
# Executar a função de inferência!
image_processed = tflite_detect_image(modelpath, imgpath, lblpath, min_conf=0.5, txt_only=False, savepath=None)

# Exibir a imagem processada usando matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image_processed)
plt.axis('off')  # Remove os eixos para uma visualização mais limpa
plt.show()