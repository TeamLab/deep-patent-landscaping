import os
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import (f1_score, precision_score, recall_score,
                                average_precision_score, roc_auc_score)


class Metrics(Callback):
    def __init__(self, mode):
        self.mode = mode
        
    def on_train_begin(self, logs=None):
        self.val_f1 = []
        self.val_recall = []
        self.val_precision = []
        self.val_average_precision = []
        self.val_auc_score = []
        
    def on_epoch_end(self, epoch, logs=None):
        if self.mode == 'all':
            val_data = [self.validation_data[0], self.validation_data[1],
                           self.validation_data[2], self.validation_data[3]]
            val_pred = np.argmax(self.model.predict(val_data), axis=1)
            val_targ = np.argmax(self.validation_data[4], axis=1)
        
        _val_f1 = f1_score(val_targ, val_pred)
        _val_recall = recall_score(val_targ, val_pred)
        _val_precision = precision_score(val_targ, val_pred)
        _val_average_precision = average_precision_score(val_targ, val_pred)
        _val_auc_score = roc_auc_score(val_targ, val_pred)
        
        self.val_f1.append(_val_f1)
        self.val_recall.append(_val_recall)
        self.val_precision.append(_val_precision)
        self.val_average_precision.append(_val_average_precision)
        self.val_auc_score.append(_val_auc_score)
        
        print("Val precision : %f, Val recall : %f, Val f1 : %f"%(_val_precision, _val_recall, _val_f1))
        print('Val average precision : %f, Val auc score : %f'%(_val_average_precision, _val_auc_score))
        
    
class Checkpoint(Callback):
    def __init__(self, metric, args, mode):
        self.metric = metric
        self.best = 0
        self.path = args.model_path
        self.name = args.data
        
    def on_epoch_end(self, epoch, logs=None):
        print('Best f1 :', self.best)
        print('Current f1 :', self.metric.val_f1[-1])
        if self.metric.val_f1[-1] > self.best:
            self.best = self.metric.val_f1[-1]
            if self.mode == 'all':
                self.model.save(os.path.join(self.path, self.name, self.name+'.model'))
        
        
        
        
        
        
        
        
