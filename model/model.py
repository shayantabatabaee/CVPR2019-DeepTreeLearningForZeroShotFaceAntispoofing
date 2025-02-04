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
DTN Model class.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import cv2
from model.utils import CRU, TRU, SFL, Conv, Downsample, Error
from model.loss import l1_loss, l2_loss, leaf_l1_loss, leaf_l2_loss, leaf_l1_score

MODEL_SAVED_PATH='./saved_model/'
WEIGHT_SAVED_PATH='./saved_weight/'

############################################################
#  Plot the figures
############################################################
def plotResults(fname, result_list):
    columm = []
    for fig in result_list:
        shape = fig.shape
        fig = fig.numpy()
        row = []
        for idx in range(shape[0]):
            item = fig[idx, :, :, :]
            if item.shape[2] == 1:
                item = np.concatenate([item, item, item], axis=2)
            item = cv2.cvtColor(cv2.resize(item, (128, 128)), cv2.COLOR_RGB2BGR)
            row.append(item)
        row = np.concatenate(row, axis=1)
        columm.append(row)
    columm = np.concatenate(columm, axis=0)
    img = np.uint8(columm * 255)
    cv2.imwrite(fname, img)


############################################################
#  Deep Tree Network
############################################################
class DTN(tf.keras.models.Model):
    def __init__(self, filters, config):
        super(DTN, self).__init__()
        self.config = config
        layer = [1, 2, 4, 8, 16]
        self.conv1 = Conv(filters, 5, apply_batchnorm=False)
        # CRU
        self.cru0 = CRU(filters)
        self.cru1 = CRU(filters)
        self.cru2 = CRU(filters)
        self.cru3 = CRU(filters)
        self.cru4 = CRU(filters)
        self.cru5 = CRU(filters)
        self.cru6 = CRU(filters)
        # TRU
        alpha = config.TRU_PARAMETERS['alpha']
        beta = config.TRU_PARAMETERS['beta']
        self.tru0 = TRU(filters, '1', alpha, beta)
        self.tru1 = TRU(filters, '2', alpha, beta)
        self.tru2 = TRU(filters, '3', alpha, beta)
        self.tru3 = TRU(filters, '4', alpha, beta)
        self.tru4 = TRU(filters, '5', alpha, beta)
        self.tru5 = TRU(filters, '6', alpha, beta)
        self.tru6 = TRU(filters, '7', alpha, beta)
        # SFL
        self.sfl0 = SFL(filters)
        self.sfl1 = SFL(filters)
        self.sfl2 = SFL(filters)
        self.sfl3 = SFL(filters)
        self.sfl4 = SFL(filters)
        self.sfl5 = SFL(filters)
        self.sfl6 = SFL(filters)
        self.sfl7 = SFL(filters)

    def call(self, inputs, training):
        x = inputs[0]
        label = inputs[1]
        if training:
            mask_spoof = label
            mask_live = 1 - label
        else:
            mask_spoof = tf.ones_like(label)
            mask_live = tf.zeros_like(label)
        ''' Tree Level 1 '''
        x = self.conv1(x, training)
        x_cru0 = self.cru0(x)
        x_tru0, route_value0, tru0_loss, mu_of_visit0, eigenvalue0, trace0 = self.tru0((x_cru0, mask_spoof), training)

        ''' Tree Level 2 '''
        x_cru00 = self.cru1(x_cru0, training)
        x_cru01 = self.cru2(x_cru0, training)
        x_tru00, route_value00, tru00_loss, mu_of_visit1, eigenvalue1, trace1 = self.tru1((x_cru00, x_tru0[0]),
                                                                                          training)
        x_tru01, route_value01, tru01_loss, mu_of_visit2, eigenvalue2, trace2 = self.tru2((x_cru01, x_tru0[1]),
                                                                                          training)

        ''' Tree Level 3 '''
        x_cru000 = self.cru3(x_cru00, training)
        x_cru001 = self.cru4(x_cru00, training)
        x_cru010 = self.cru5(x_cru01, training)
        x_cru011 = self.cru6(x_cru01, training)
        x_tru000, route_value000, tru000_loss, mu_of_visit3, eigenvalue3, trace3 = self.tru3((x_cru000, x_tru00[0]),
                                                                                             training)
        x_tru001, route_value001, tru001_loss, mu_of_visit4, eigenvalue4, trace4 = self.tru4((x_cru001, x_tru00[1]),
                                                                                             training)
        x_tru010, route_value010, tru010_loss, mu_of_visit5, eigenvalue5, trace5 = self.tru5((x_cru010, x_tru01[0]),
                                                                                             training)
        x_tru011, route_value011, tru011_loss, mu_of_visit6, eigenvalue6, trace6 = self.tru6((x_cru011, x_tru01[1]),
                                                                                             training)

        ''' Tree Level 4 '''
        map0, cls0 = self.sfl0(x_cru000, training)
        map1, cls1 = self.sfl1(x_cru000, training)
        map2, cls2 = self.sfl2(x_cru001, training)
        map3, cls3 = self.sfl3(x_cru001, training)
        map4, cls4 = self.sfl4(x_cru010, training)
        map5, cls5 = self.sfl5(x_cru010, training)
        map6, cls6 = self.sfl6(x_cru011, training)
        map7, cls7 = self.sfl7(x_cru011, training)
        ''' Output '''
        maps = [map0, map1, map2, map3, map4, map5, map6, map7]
        clss = [cls0, cls1, cls2, cls3, cls4, cls5, cls6, cls7]
        route_value = [route_value0, route_value00, route_value01,
                       route_value000, route_value001, route_value010, route_value011]
        x_tru0000 = tf.concat([x_tru000[0], mask_live], axis=1)
        x_tru0001 = tf.concat([x_tru000[1], mask_live], axis=1)
        x_tru0010 = tf.concat([x_tru001[0], mask_live], axis=1)
        x_tru0011 = tf.concat([x_tru001[1], mask_live], axis=1)
        x_tru0100 = tf.concat([x_tru010[0], mask_live], axis=1)
        x_tru0101 = tf.concat([x_tru010[1], mask_live], axis=1)
        x_tru0110 = tf.concat([x_tru011[0], mask_live], axis=1)
        x_tru0111 = tf.concat([x_tru011[1], mask_live], axis=1)
        leaf_node_mask = [x_tru0000, x_tru0001, x_tru0010, x_tru0011, x_tru0100, x_tru0101, x_tru0110, x_tru0111]

        if training:
            # for the training
            route_loss = [tru0_loss[0], tru00_loss[0], tru01_loss[0],
                          tru000_loss[0], tru001_loss[0], tru010_loss[0], tru011_loss[0]]
            recon_loss = [tru0_loss[1], tru00_loss[1], tru01_loss[1],
                          tru000_loss[1], tru001_loss[1], tru010_loss[1], tru011_loss[1]]
            mu_update = [mu_of_visit0,
                         mu_of_visit1,
                         mu_of_visit2,
                         mu_of_visit3,
                         mu_of_visit4,
                         mu_of_visit5,
                         mu_of_visit6]
            eigenvalue = [eigenvalue0,
                          eigenvalue1,
                          eigenvalue2,
                          eigenvalue3,
                          eigenvalue4,
                          eigenvalue5,
                          eigenvalue6]
            trace = [trace0,
                     trace1,
                     trace2,
                     trace3,
                     trace4,
                     trace5,
                     trace6]

            return maps, clss, route_value, leaf_node_mask, [route_loss, recon_loss], mu_update, eigenvalue, trace
        else:
            return maps, clss, route_value, leaf_node_mask


class Model:
    def __init__(self, config):
        self.config = config
        # model
        self.dtn = DTN(32, config)
        # model optimizer
        self.dtn_op = tf.compat.v1.train.AdamOptimizer(config.LEARNING_RATE, beta1=0.5)
        # model losses
        self.depth_map_loss = Error()
        self.class_loss = Error()
        self.route_loss = Error()
        self.uniq_loss = Error()
        # model saving setting
        self.last_epoch = 0
        self.checkpoint_manager = []

    def compile(self, checkpoint_num=None):
        checkpoint_dir = self.config.LOG_DIR
        checkpoint = tf.train.Checkpoint(dtn=self.dtn,
                                         dtn_optimizer=self.dtn_op)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=60)
        last_checkpoint = self.checkpoint_manager.latest_checkpoint
        if checkpoint_num is not None:
            last_checkpoint = last_checkpoint.split('-')[0] + '-' + str(checkpoint_num)
        checkpoint.restore(last_checkpoint)
        if last_checkpoint:
            self.last_epoch = int(last_checkpoint.split('-')[-1])
            print("Restored from {}".format(last_checkpoint))
        else:
            print("Initializing from scratch.")

    def train(self, train, val=None):
        config = self.config
        step_per_epoch = config.STEPS_PER_EPOCH
        step_per_epoch_val = config.STEPS_PER_EPOCH_VAL
        epochs = config.MAX_EPOCH

        # data stream
        it = train.get_feed()
        global_step = self.last_epoch * step_per_epoch
        if val is not None:
            it_val = val.get_feed()
        for epoch in range(self.last_epoch, epochs):
            start = time.time()
            # define the
            self.dtn_op = tf.compat.v1.train.AdamOptimizer(config.LEARNING_RATE, beta1=0.5)
            ''' train phase'''
            for step in range(step_per_epoch):
                depth_map_loss, class_loss, route_loss, uniq_loss, spoof_counts, eigenvalue, trace, _to_plot =\
                    self.train_one_step(next(it), global_step, True)
                # display loss
                global_step += 1
                print('Epoch {:d}-{:d}/{:d}: Map:{:.3g}, Cls:{:.3g}, Route:{:.3g}({:3.3f}, {:3.3f}), Uniq:{:.3g}, '
                      'Counts:[{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}]     '.
                      format(epoch + 1, step + 1, step_per_epoch,
                             self.depth_map_loss(depth_map_loss),
                             self.class_loss(class_loss),
                             self.route_loss(route_loss), eigenvalue, trace,
                             self.uniq_loss(uniq_loss),
                             spoof_counts[0], spoof_counts[1], spoof_counts[2], spoof_counts[3],
                             spoof_counts[4], spoof_counts[5], spoof_counts[6], spoof_counts[7],), end='\r')
                # plot the figure
                '''if (step + 1) % 400 == 0:
                    fname = self.config.LOG_DIR + '/epoch-' + str(epoch + 1) + '-train-' + str(step + 1) + '.png'
                    plotResults(fname, _to_plot)'''

            # save the model
            if (epoch + 1) % 1 == 0:
                self.checkpoint_manager.save(checkpoint_number=epoch + 1)
            print('\n', end='\r')

            ''' eval phase'''
            if val is not None:
                for step in range(step_per_epoch_val):
                    depth_map_loss, class_loss, route_loss, uniq_loss, spoof_counts, eigenvalue, trace, _to_plot =\
                        self.train_one_step(next(it_val), global_step, False)
                    # display something
                    print('    Val-{:d}/{:d}: Map:{:.3g}, Cls:{:.3g}, Route:{:.3g}({:3.3f}, {:3.3f}), Uniq:{:.3g}, '
                          'Counts:[{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}]     '.
                          format(step + 1, step_per_epoch_val,
                                 self.depth_map_loss(depth_map_loss, val=1),
                                 self.class_loss(class_loss, val=1),
                                 self.route_loss(route_loss, val=1), eigenvalue, trace,
                                 self.uniq_loss(uniq_loss, val=1),
                                 spoof_counts[0], spoof_counts[1], spoof_counts[2], spoof_counts[3],
                                 spoof_counts[4], spoof_counts[5], spoof_counts[6], spoof_counts[7], ), end='\r')
                    # plot the figure
                    '''if (step + 1) % 100 == 0:
                        fname = self.config.LOG_DIR + '/epoch-' + str(epoch + 1) + '-val-' + str(step+1) + '.png'
                        plotResults(fname, _to_plot)'''
                self.depth_map_loss.reset()
                self.class_loss.reset()
                self.route_loss.reset()
                self.uniq_loss.reset()

            # time of one epoch
            print('\n    Time taken for epoch {} is {:3g} sec'.format(epoch + 1, time.time() - start))
        return 0

    def evaluate(self, val):
        config = self.config
        step_per_epoch_val = config.STEPS_PER_EPOCH_VAL

        y = np.zeros((0, 1))
        y_hat = np.zeros((0, 1))

        if val is not None:
            it_val = val.get_feed()
            start = time.time()
            self.dtn_op = tf.compat.v1.train.AdamOptimizer(config.LEARNING_RATE, beta1=0.5)
            ''' eval phase'''
            for step in range(step_per_epoch_val):
                image, dmap, labels = next(it_val)
                dmap_pred, cls_pred, route_value, leaf_node_mask = self.dtn([image, labels], False)
                cls = np.asarray(cls_pred)[:, :, 0].T
                leaf = np.asarray(leaf_node_mask)[:, :, 0].T
                _y_hat = np.sum(leaf * cls, axis=1)[:, np.newaxis]
                _y = np.asarray(labels)
                y = np.vstack((y, _y))
                y_hat = np.vstack((y_hat, _y_hat))
            # time of evaluation
            print('\n    Time taken for evaluation is {:3g} sec'.format(time.time() - start))
            np.seterr(divide='warn')

        return y, y_hat

    def train_one_step(self, data_batch, step, training):
        dtn = self.dtn
        dtn_op = self.dtn_op
        image, dmap, labels = data_batch
        with tf.GradientTape() as tape:
            dmap_pred, cls_pred, route_value, leaf_node_mask, tru_loss, mu_update, eigenvalue, trace =\
                dtn([image, labels], True)

            # supervised feature loss
            depth_map_loss = leaf_l1_loss(dmap_pred, tf.image.resize(dmap, [32, 32]), leaf_node_mask)
            class_loss = leaf_l1_loss(cls_pred, labels, leaf_node_mask)
            supervised_loss = depth_map_loss + 0.001*class_loss

            # unsupervised tree loss
            route_loss = tf.reduce_mean(tf.stack(tru_loss[0], axis=0) * [1., 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            uniq_loss  = tf.reduce_mean(tf.stack(tru_loss[1], axis=0) * [1., 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            eigenvalue = np.mean(np.stack(eigenvalue, axis=0) * [1., 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            trace = np.mean(np.stack(trace, axis=0) * [1., 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            unsupervised_loss = 2*route_loss + 0.001*uniq_loss

            # total loss
            if step > 10000:
                loss = supervised_loss + unsupervised_loss
            else:
                loss = supervised_loss

        if training:
            # back-propagate
            gradients = tape.gradient(loss, dtn.variables)
            dtn_op.apply_gradients(zip(gradients, dtn.variables))

            # Update mean values for each tree node
            mu_update_rate = self.config.TRU_PARAMETERS["mu_update_rate"]
            mu = [dtn.tru0.project.mu, dtn.tru1.project.mu, dtn.tru2.project.mu, dtn.tru3.project.mu,
                  dtn.tru4.project.mu, dtn.tru5.project.mu, dtn.tru6.project.mu]
            for mu, mu_of_visit in zip(mu, mu_update):
                if step == 0:
                    update_mu = mu_of_visit
                else:
                    update_mu = mu_of_visit * mu_update_rate + mu * (1 - mu_update_rate)
                K.set_value(mu, update_mu)

        # leaf counts
        spoof_counts = []
        for leaf in leaf_node_mask:
            spoof_count = tf.reduce_sum(leaf[:, 0]).numpy()
            spoof_counts.append(int(spoof_count))

        _to_plot = [image, dmap, dmap_pred[0]]

        return depth_map_loss, class_loss, route_loss, uniq_loss, spoof_counts, eigenvalue, trace, _to_plot

    def save(self):
        x = np.ones((1, 256, 256, 3), dtype=np.float32)
        label = np.ones((1, 1), dtype=np.float32)
        # y = self.dtn((x, np.ones((1, 1), dtype=np.float32)), training=False)
        # self.dtn.save('./save/')
        self.dtn._set_inputs((x, label), training=False)
        os.makedirs(MODEL_SAVED_PATH)
        tf.saved_model.save(self.dtn, MODEL_SAVED_PATH)

    def load(self):
        self.dtn = tf.saved_model.load(MODEL_SAVED_PATH)

    def save_weight(self):
        x = np.ones((1, 256, 256, 3), dtype=np.float32)
        label = np.ones((1, 1), dtype=np.float32)
        y = self.dtn((x, label), training=False)
        os.makedirs(WEIGHT_SAVED_PATH)
        self.dtn.save_weights(WEIGHT_SAVED_PATH)

    def load_weight(self):
        self.dtn.load_weights(WEIGHT_SAVED_PATH)
