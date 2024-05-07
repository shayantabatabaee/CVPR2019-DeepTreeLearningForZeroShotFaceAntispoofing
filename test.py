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

import numpy as np
import matplotlib.pyplot as plt

from metrics import Metrics
from model.dataset import Dataset
from model.config import Config
from model.model import Model
from math import ceil

def main(argv=None):
    # Configurations
    config = Config()
    config.DATA_DIR = './data/'
    #config.DATA_DIR_VAL = './data_laplacian_300/mehrdad'
    config.DATA_DIR_VAL = "/media/disk-2/fold1/validation/"
    config.LOG_DIR = './log/model'
    config.MODE = 'validation'

    dataset_validation = Dataset(config, 'validation')
    config.STEPS_PER_EPOCH_VAL = dataset_validation.get_dataset_count() // config.BATCH_SIZE

    config.display()
    # Build a Graph
    model = Model(config)
    # model.load()

    results = []
    ranges = range(1, 61)
    # ranges = [60]
    for i in ranges:
        model.compile(i)
        y, y_hat = model.evaluate(dataset_validation)
        # np.save('fold2_val_confidences.npy', np.asarray(y_hat))
        # loss = -np.mean(np.nan_to_num(np.log(y_hat)) * np.asarray(y) +
        #                 np.nan_to_num(np.log(1 - y_hat)) * (1 - np.asarray(y)))
        loss = np.mean(np.power(y_hat - y, 2), axis=0)
        confusion_matrix = Metrics.confusion_matrix(y, y_hat, 0.1)
        precision = Metrics.precision(confusion_matrix)
        recall = Metrics.recall(confusion_matrix)
        accuracy = Metrics.accuracy(confusion_matrix)
        f1score = Metrics.f_score(precision, recall)
        far, frr = Metrics.error_rate(confusion_matrix)
        apcer = Metrics.apcer(y, y_hat)
        bpcer = Metrics.bpcer(y, y_hat)
        acer = Metrics.acer(apcer, bpcer)
        hter = Metrics.hter(far, frr)

        print(f"Total data count: {np.sum(confusion_matrix)}")

        # fpr, tpr, auc_value, dist, eer = Metrics.roc_values(y, y_hat)
        # Metrics.plot_roc(fpr, tpr, auc_value, dist, eer)
        # plt.show()

        # Metrics.plot_pr(y, y_hat)
        #Metrics.plot_histogram(y, y_hat, i)
        # fpr, tpr, auc_value, dist, eer = Metrics.roc_values(y, y_hat)
        # Metrics.plot_roc(fpr, tpr, auc_value, dist, eer)
        print('loss: %f \naccuracy: %f \nprecision: %f \nrecall: %f \nf-score: %f'
              % (loss, accuracy, precision, recall, f1score))
        print(
            f'tp: {confusion_matrix[3]}, '
            f'tn: {confusion_matrix[0]}, '
            f'fp: {confusion_matrix[1]}, '
            f'fn: {confusion_matrix[2]}'
            f'\nfar: {far:.3f}, '
            f'frr: {frr:.3f}, '
            f'apcer: {apcer:.3f}, '
            f'bpcer: {bpcer:.3f}, '
            f'acer: {acer:.3f}, '
            f'hter: {hter:.3f} ')
        # plt.show()

        results.append((loss, accuracy, precision, recall, f1score))

    # accuracies = [result[1] for result in results]
    # precisions = [result[2] for result in results]
    # recalls = [result[3] for result in results]
    # f1scores = [result[4] for result in results]
    # Metrics.plot_evaluate_results(precisions, recalls, f1scores, accuracies, range(51, 101))


if __name__ == '__main__':
    main()
