from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import request

import numpy as np
import tensorflow as tf
import base64
import cv2


MODEL_FILE = "./Model/detect_float16.tflite"
#OUTPUT_IMAGE = "Output.jpg"
#IMAGE_FILE_PATH="./Model/4.jpg"
VERBOSE = False
mean = 127.5
std = 255.0

interpreter = tf.lite.Interpreter(MODEL_FILE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if VERBOSE:
    print(input_details)
    print(output_details)


floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


dims = (width, height)


def decode_from_b64(b64string):
    im_b64 = base64.b64decode(b64string)
    im_arr = np.frombuffer(im_b64, dtype=np.uint8)
    img = cv2.imdecode(im_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    img = cv2.resize(img, dims)
    #print(img.shape)
    return img,height,width,channels

app= Flask(__name__)

@app.route('/getsquare2',methods=['POST'])
def index():
    solicitud=request.get_json()
    square,height,width,channels = decode_from_b64(solicitud['square'])
    input_data = np.expand_dims(square, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - mean) / std
   # print(input_data.shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    detection_boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = interpreter.get_tensor(output_details[3]['index'])
    classe=classes[0]
    score= scores[0]
    ymin = 0
    xmin = 0
    ymax = 0
    xmax = 0
   # ar=0
   # h=0
   # w=0
    #print(detection_boxes)
    #print(classes)
    #print(scores)
    #print results
    #img = Image.open(IMAGE_FILE_PATH)
    for count,bbox in enumerate(detection_boxes[0]):
        #width, height = img.size

        y_min = int(bbox[0] * height)
        x_min = int(bbox[1] * width)
        y_max = int(bbox[2] * height)
        x_max = int(bbox[3] * width)
        #h=0
        #w=0
        #ar=0
        if(score[count]>.7):
            #if(classe[count] % 1 ==0 and classe[count]>0):
            #h=(y_max-y_min)
            #w=(x_max-x_min)
            #if(h > 0 and w > 0):
            #    ar=h/w
            #    print(ar) and ( ar > 0.59 and ar <0.685)
            if ((((y_max-y_min)*(x_max-x_min))>((ymax-ymin)*(xmax-xmin)))):
                xmin=x_min
                ymin=y_min
                xmax=x_max
                ymax=y_max
                    #draw = ImageDraw.Draw(img)
                    #draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="#0F0")
    print(xmin)
    print(ymin)
    print(xmax)
    print(ymax)
   # img.save(OUTPUT_IMAGE)
    return {"xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax }


app.run(host = 'localhost',port=65535)
