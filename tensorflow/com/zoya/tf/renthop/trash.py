'''
Created on Apr 1, 2017

@author: zoya
'''
import os
import pandas
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn


TRAIN_DF = 'train_184.csv'
TEST_DF = 'test_184.csv'

TF_TRAIN_DF = 'tf_train_184.csv'
TF_TEST_DF = 'tf_test_184.csv'


def clean_df():
    os.chdir('/Users/zoya/projects/datascience/tensorflow/tensorflow/com/zoya/tf/renthop/')
    train=pandas.read_csv(TRAIN_DF)
    categorical_vars = ['building_id', 'manager_id', 'listing_id', 'street_address', "display_address", 'display_address_cleaned']
    continues_vars = list(set(train.columns.values[1:20]) - set(categorical_vars))
    X = train[categorical_vars + continues_vars]
    for var in categorical_vars:
        le = LabelEncoder().fit(X[var])
        X[var + '_ids'] = le.transform(X[var])
        X.pop(var)
    
    for var in categorical_vars:
        if (var in X): 
            print("Was unable to factorize column: " + var)
            X.drop(var, 1, inplace=True)
    
    X.to_csv(TF_TRAIN_DF, index=False)


def pandas_input_fn(x, y=None, batch_size=128, num_epochs=None):
    def input_fn():
        if y is not None:
            x['target'] = y
        queue = None#learn.dataframe.queues.feeding_functions.enqueue_data(x, 1000, shuffle=num_epochs is None, num_epochs=num_epochs)
        if num_epochs is None:
            features = queue.dequeue_many(batch_size)
        else:
            features = queue.dequeue_up_to(batch_size)
        features = dict(zip(['index'] + list(x.columns), features))
        if y is not None:
            target = features.pop('target')
            return features, target
        return features
    
    return input_fn
   
    
    #final_features = [tf.expand_dims(tf.cast(features[var], tf.float32), 1) for var in continues_vars]

