# Copyright 2018
# 
# Yaojie Liu, Amin Jourabloo, Xiaoming Liu, Michigan State University
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from datetime import datetime
import time

import tensorflow as tf
from model.dataset import Dataset
from model.config import Config
from model.model import Model


def main(argv=None):
    # Configurations
    config = Config()
    config.DATA_DIR = './data/'
    config.DATA_DIR_VAL = './data_val/'
    config.LOG_DIR = './log/model'

    # Get images and labels.
    dataset_train = Dataset(config, 'train')
    config.STEPS_PER_EPOCH = dataset_train.get_dataset_count() // config.BATCH_SIZE

    dataset_validation = Dataset(config, 'validation')
    config.STEPS_PER_EPOCH_VAL = dataset_validation.get_dataset_count() // config.BATCH_SIZE
    config.display()

    # Build a Graph
    model = Model(config)
    # Train the model
    model.compile()
    model.train(dataset_train, dataset_validation)


if __name__ == '__main__':
    main()
