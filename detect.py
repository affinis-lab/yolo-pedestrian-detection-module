from yolo import Yolo
from utils import decode_netout, draw_boxes, decode_annot_netout, draw_anots
from keras.models import load_model
import numpy as np
import json
import cv2
from constants import *
import time
import glob
from os.path import join
import tensorflow as tf
import matplotlib.pyplot as plt


def predict_image(image, model):

    scaled_image = cv2.resize(image, (416, 416))
    input_image = scaled_image / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)

    start = time.time()

    dummy_array = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))

    netout = model.predict([input_image, dummy_array])
    #netout = model.predict(input_image)
    netout = netout[0]
    number_of_classes = NUM_CLASSES
    #number_of_classes = 80
    netout = np.reshape(netout, (VERTICAL_GRIDS, HORIZONTAL_GRIDS, BOX, 4 + 1 + number_of_classes))

    print(netout.shape)

    end = time.time()

    print("Prediction took " + str(end-start) + " seconds.")

    start = time.time()
    boxes = decode_netout(netout,
                          obj_threshold=OBJ_THRESHOLD,
                          nms_threshold=NMS_THRESHOLD,
                          anchors=ANCHORS,
                          nb_class=NUM_CLASSES)

    end = time.time()

    print("Filtering and NMS took " + str(end-start) + " seconds.")

    # image = draw_anots(input_image, boxes, labels = LABELS)

    image = draw_boxes(image, boxes, labels=LABELS)
    end1 = time.time()

    print("Drawing boxes took " + str(end1-end) + " seconds.")

    plt.imshow(image[:, :, ::-1])
    plt.show()


def dummy_loss(y_true, y_pred):
    return y_pred


if __name__ == "__main__":

    CONFIG_FILE = 'config.json'
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    model_name = "fYoloLastFull-25-30"
    model_path = "models/" + model_name + ".h5"
    yolo = Yolo(config)
    model = load_model(model_path, custom_objects={'custom_loss_basara': dummy_loss, 'tf': tf})

    files = []
    for ext in ('*.jpg', '*.png'):
        files.extend(glob.glob(join("./test", ext)))

    total_gueses = 0

    brojac = -1

    for image_path in files:
        brojac += 1
        im = cv2.imread(image_path)
        predict_image(im, model)
