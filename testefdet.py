# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 07:17:21 2020

@author: Abhinav
"""
from model_utils import detect_fn, load_image_into_numpy_array, configs
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
import os
import glob

import random
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

TEST_IMAGE_PATHS = glob.glob('./PERCEPTION/raw/*.jpg')
# image_path = random.choice(TEST_IMAGE_PATHS)

def detect(image_nps, fname=None, persist=True):
    targets = []
    for image_np in image_nps:
        # image_np = load_image_into_numpy_array(image_path)            
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)
        
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        
        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'][0].numpy(),
              (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
              detections['detection_scores'][0].numpy(),
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=.5,
              agnostic_mode=False,
        )
        # print(detections['detection_scores'])
        scores = detections['detection_scores'][0]
        dimensions =  detections['detection_boxes'][0]
        classes = detections['detection_classes'][0]
        for i in range(scores.shape[0]):
            if(scores[i].numpy() > 0.63):
                if(classes[i].numpy() != 2):
                    ymin = dimensions[i].numpy()[0]
                    xmin = dimensions[i].numpy()[1]
                    ymax = dimensions[i].numpy()[2]
                    xmax = dimensions[i].numpy()[3]
                    x = (xmin+xmax)*image_np.shape[1]/2
                    y = (ymin+ymax)*image_np.shape[0]/2
                    targets.append((x,y))
                    print(classes[i].numpy())        
        if(persist == True):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S.%f")[:-4]
            num = current_time.split(':')
            num = '_'.join(num)
            fname = './PERCEPTION/detection/img' + num + '.jpg'
            img = Image.fromarray(image_np_with_detections, 'RGB')   
            img.save(fname)

    return targets