# Copyright 2019
#
# Yaojie Liu, Joel Stehouwer, Amin Jourabloo, Xiaoming Liu, Michigan State University
#
# All Rights Reserved.
#
# This research is based upon work supported by the Office of the Director of
# National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
# (IARPA), via IARPA R&D Contract No. 2017-17020200004. The views and
# conclusions contained herein are those of the authors and should not be
# interpreted as necessarily representing the official policies or endorsements,
# either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The
# U.S. Government is authorized to reproduce and distribute reprints for
# Governmental purposes not withstanding any copyright annotation thereon.
# ==============================================================================
"""
DTN for Zero-shot Face Anti-spoofing
Data Loading class.

"""
import tensorflow as tf
import numpy as np
import os
import random


class Dataset:

    def __init__(self, config, mode):
        self.config = config
        is_shuffling = False
        if mode == 'train':
            data_dir = self.config.DATA_DIR
            is_shuffling = True
        elif mode == 'validation':
            data_dir = self.config.DATA_DIR_VAL
        elif mode == 'test':
            data_dir = self.config.DATA_DIR_TEST
        else:
            raise Exception("Invalid Mode, acceptable modes are: train, validation, test")

        self.__features, self.__labels = self.__load_files(data_dir)
        self.__dataset_count = len(self.__features)
        self.__input_tensors = self.inputs_for_training(self.config, is_shuffling)
        self.__feed = iter(self.__input_tensors)

    def get_dataset_count(self):
        return self.__dataset_count

    def get_feed(self):
        return self.__feed

    def inputs_for_training(self, config, is_shuffling):
        def generator():
            for i in range(0, self.__dataset_count):
                yield self.__features[i], self.__labels[i]

        dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.uint8),
                                                 output_shapes=([config.IMAGE_SIZE, config.IMAGE_SIZE, 3], ()),
                                                 args=[])
        if is_shuffling:
            dataset = dataset.shuffle(self.config.BATCH_SIZE, reshuffle_each_iteration=True)
        dataset = dataset.repeat(-1)
        dataset = dataset.map(map_func=self.__parse_fn)
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    @staticmethod
    def __load_files(path):
        features = []
        labels = []
        for key in os.listdir(path):
            for file_name in os.listdir(os.path.join(path, key)):
                if 'dataset_' in file_name:
                    file = np.load(os.path.join(path, key, file_name), allow_pickle=True)
                    data = file[:, 1]
                    label = file[:, 0]
                    if key == 'mobile_distance':
                        data = data[np.where(label == 1)]
                        label = label[np.where(label == 1)]
                    features.extend(data)
                    label = -label + 1
                    labels.extend(label)
        temp_array = list(zip(features, labels))
        random.Random(0).shuffle(temp_array)
        features, labels = zip(*temp_array)
        return features, labels

    def __parse_fn(self, X, Y):
        config = self.config
        dmap_size = config.MAP_SIZE

        def _parse_function(x, y):
            image = x / 255
            dmap = None
            if y == 1:
                # Spoof
                dmap = np.ones((dmap_size, dmap_size, 1), dtype=np.float32)
            elif y == 0:
                # Live
                dmap = np.zeros((dmap_size, dmap_size, 1), dtype=np.float32)

            label = np.reshape(y, (1,))
            dmap1 = dmap * (1-label)
            dmap2 = np.ones_like(dmap) * label
            dmap = np.concatenate([dmap1, dmap2], axis=2)
            return image.astype(np.float32), dmap.astype(np.float32), label.astype(np.float32)

        image_ts, dmap_ts, label_ts = tf.numpy_function(_parse_function, [X, Y], [tf.float32, tf.float32, tf.float32])
        image_ts = tf.ensure_shape(image_ts, [config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
        dmap_ts  = tf.ensure_shape(dmap_ts,  [config.MAP_SIZE, config.MAP_SIZE, 2])
        label_ts = tf.ensure_shape(label_ts, [1])
        return image_ts, dmap_ts, label_ts
