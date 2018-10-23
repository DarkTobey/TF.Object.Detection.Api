# -*- coding: utf-8 -*-
"""
Usage:
  # 生成tensorflow训练所需的 TF Record 文件
  # From tensorflow/models/
"""

import os
import io
import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from PIL import Image
from collections import namedtuple, OrderedDict

fullpath = 'D:/Tools/TensorFlow/models/research/object_detection'
os.chdir(fullpath)


base_dir = "D:/Tools/Train/dnf"
label_map_path = base_dir + "/config/label_map.pbtxt"

train_csv_path = base_dir + "/data/train_label.csv"
train_output_path = base_dir + "/data/train.record"

eval_csv_path = base_dir + "/data/eval_label.csv"
eval_output_path = base_dir + "/data/eval.record"


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, label_map_dict):
    with tf.gfile.GFile(group.filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map_dict[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def read_and_create(label_map_dict, csv_path, out_path):
    writer = tf.python_io.TFRecordWriter(out_path)
    examples = pd.read_csv(csv_path)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(out_path))


def main(_):
    # 所有标签
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    # 生成train record
    read_and_create(label_map_dict, train_csv_path, train_output_path)

    # 生成eval record
    read_and_create(label_map_dict, eval_csv_path, eval_output_path)


if __name__ == '__main__':
    tf.app.run()
