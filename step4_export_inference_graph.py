# -*- coding: utf-8 -*-
"""
Usage:
    # 根据训练结果生成推理图，即模型文件
    python object_detection/model_main.py
        --pipeline_config_path=data/train/ssd_mobilenet_v1_coco.config
        --model_dir=data/output
        --num_train_steps=50000
        --num_eval_steps=2000
        --alsologtostderr
"""

import tensorflow as tf
import os
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2


fullpath = 'D:/Tools/TensorFlow/models/research/object_detection'
os.chdir(fullpath)


base_dir = "D:/Tools/Train/dnf"
pipeline_path = base_dir + '/config/ssd_mobilenet_v1_coco.config'
out_train_ckpt_file = base_dir + '/output/model.ckpt-194911'
out_model_path = base_dir + '/output/result'


slim = tf.contrib.slim
flags = tf.app.flags
flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`]')
flags.DEFINE_string('input_shape', None,
                    'If input_type is `image_tensor`, this can explicitly set '
                    'the shape of this input tensor to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. If not specified, for an `image_tensor, the '
                    'default shape will be partially specified as '
                    '`[None, None, None, 3]`.')
flags.DEFINE_string('pipeline_config_path', pipeline_path,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_prefix', out_train_ckpt_file,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', out_model_path,
                    'Path to write outputs.')
flags.DEFINE_string('config_override', '',
                    'pipeline_pb2.TrainEvalPipelineConfig '
                    'text proto to override pipeline_config_path.')
flags.DEFINE_boolean('write_inference_graph', False,
                     'If true, writes inference graph to disk.')
tf.app.flags.mark_flag_as_required('pipeline_config_path')
tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
tf.app.flags.mark_flag_as_required('output_directory')
FLAGS = flags.FLAGS


def main(_):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(FLAGS.config_override, pipeline_config)
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in FLAGS.input_shape.split(',')
        ]
    else:
        input_shape = None
    exporter.export_inference_graph(
        FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_prefix,
        FLAGS.output_directory, input_shape=input_shape,
        write_inference_graph=FLAGS.write_inference_graph)


if __name__ == '__main__':
    tf.app.run()
