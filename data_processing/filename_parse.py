#!/usr/bin/python
# encoding: utf-8
from data_processing.fileread_util import txt_parse

def deepfashion_name_parse(f, mode='train'):
    if mode == 'train':
        data_type = 'train'
    elif mode == 'val':
        data_type = 'gallery'
    elif mode == 'test':
        data_type = 'query'
    lines = txt_parse(f)
    num_train = 0
    result = []
    for line in lines:
        if line[0:4] != 'img/':
            continue
        # print(len(line.split()))
        if len(line.split()) > 0 and line.split()[2] == data_type:
            num_train += 1
            name = line.split()[0]
            img_class = name.split('/')[2]
            result.append(img_class + ' ' + name)
    print("The first image class and name are {}".format(result[0]))
    print("The number of images is {}".format(len(result)))
    return result
