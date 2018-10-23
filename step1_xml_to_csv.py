# -*- coding: utf-8 -*-
"""
Usage:
  # 给训练数据打上标签
  # 修改 fullpath 的值 , 用于指定训练数据所在位置
  # 直接执行 ,  即可生成csv文件
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


base_dir = 'D:/Tools/Train/dnf'
save_path = base_dir + '/data'

train_path = base_dir + '/data/train'
eval_path = base_dir + '/data/eval'


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (path + '/' + root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    os.chdir(train_path)
    xml_train = xml_to_csv(train_path)
    xml_train.to_csv(save_path + '/train_label.csv', index=None)
    print('Successfully converted train csv.')

    os.chdir(eval_path)
    xml_eval = xml_to_csv(eval_path)
    xml_eval.to_csv(save_path + '/eval_label.csv', index=None)
    print('Successfully converted eval csv.')


main()
