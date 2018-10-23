# -*- coding: utf-8 -*-
"""
Usage:
  # 利用刚才训练得到的模型进行目标检测
  # From tensorflow/models/
"""

import time
import numpy as np
import os
import glob
import shutil
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

fullpath = 'D:/Tools/TensorFlow/models/research/object_detection'
os.chdir(fullpath)

sys.path.append("..")


# PATH_TO_CKPT = "D:/Tools/TensorFlow/pb/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb"
# PATH_TO_LABELS = "D:/Tools/TensorFlow/pb/ssd_mobilenet_v1_coco_2018_01_28/mscoco_label_map.pbtxt"


# 模型文件位置和输出位置
base_dir = "D:/Tools/Train/dnf"
PATH_TO_CKPT = base_dir + "/output/result/frozen_inference_graph.pb"
PATH_TO_LABELS = base_dir + "/config/label_map.pbtxt"

test_image_path = base_dir + '/data/test'
output_image_path = base_dir + '/data/result'
output_csv_path = base_dir + '/data/result'


# 加载模型文件
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# 加载label
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=99, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# rgb图像 转 科学计算用的数组 （现在只能转jpg格式图像，png图像太大了）
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# 开始检测
start = time.time()
os.chdir(test_image_path)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # 获取图的输出
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name(
            'detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name(
            'num_detections:0')

        # 删除旧的保存文件
        if os.path.exists(output_image_path):
            shutil.rmtree(output_image_path)
        os.makedirs(output_image_path)

        data = pd.DataFrame()
        files = os.listdir(test_image_path)
        for fileName in files:
            if(fileName[-3:] == 'jpg'):
                # 读取图片并检测
                image = Image.open(fileName)
                width, height = image.size
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # 画上结果
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3)
                image_save = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_image_path + '/' + fileName, image_save)

                s_boxes = boxes[scores > 0.5]
                s_classes = classes[scores > 0.5]
                s_scores = scores[scores > 0.5]

                # 保存位置坐标结果到 .csv表格
                for i in range(len(s_classes)):
                    newdata = pd.DataFrame(0, index=range(1), columns=range(7))
                    newdata.iloc[0, 0] = fileName
                    newdata.iloc[0, 1] = s_boxes[i][0]*height  # ymin
                    newdata.iloc[0, 2] = s_boxes[i][1]*width   # xmin
                    newdata.iloc[0, 3] = s_boxes[i][2]*height  # ymax
                    newdata.iloc[0, 4] = s_boxes[i][3]*width   # xmax
                    newdata.iloc[0, 5] = s_scores[i]
                    newdata.iloc[0, 6] = s_classes[i]
                    data = data.append(newdata)
                data.to_csv(output_csv_path + '/' + 'result.csv', index=False)

end = time.time()
print("Execution Time: ", end - start)
