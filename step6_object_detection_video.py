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

test_video_path = base_dir + '/data/test/video.mp4'
test_video_save_path = base_dir + '/data/result/video_save.mp4'


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

        # 读取测试视频
        vidcap = cv2.VideoCapture(test_video_path)
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))

        out_video = cv2.VideoWriter(test_video_save_path, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

        while(True):
            ret, image = vidcap.read()
            if ret == True:
                # 读取每一帧
                image_np = image
                # image_np = load_image_into_numpy_array(image)

                # 图像处理后 放入session进行检测
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

                # 创建窗口并显示
                # plt.figure(figsize=())
                # plt.imshow(image_np)
                cv2.imshow('video', image_np)

                # 保存
                # image_save = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                out_video.write(image_np)

        # Break the loop
            else:
                print("read video error")
                break


end = time.time()
print("Execution Time: ", end - start)
