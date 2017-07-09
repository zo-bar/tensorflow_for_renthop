'''
Created on Apr 5, 2017

@author: zoya
'''
import os
import pandas
import tensorflow as tf
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split


TRAIN_DF = 'train_184.csv'
TEST_DF = 'test_183.csv'

LABEL_COLUMN = "out"
LINEAR_CAT_COLUMNS = ['building_id', 'manager_id', 'street_address',
                       'display_address', 'display_address_cleaned']
LINEAR_CONT_COLUMNS = []
DNN_CONT_COLUMNS = ['price_ratio', 'bed_price', 'price', 'bed_price_desc', 'feature_count',
                      'photo_count', 'weekday', 'fee', 'hour', 'mean_bed_price',
                      'bed_ratio', 'bed_ratio_desc', 'bedrooms_desc', 'location', 'latitude', 'longitude',
                      'long_lat_185', 'walk', 'ok', 'long_lat_339',
                      'long_lat_126', 'renovated', 'super',
                      'level', 'internet', 'pets',
                      'exclusive', 'long_lat_336', 'public_laundry', 'luxury', 'long_lat_362', 'central',
                      'room_price_desc', 'long_lat_52', 'access', 'newly',
                      'long_lat_333', 'long_lat_317', 'loft', 'long_lat_11',
                      'long_lat_107', 'granite', 'space', 'old', 'long_lat_6',
                      'long_lat_373', 'long_lat_105', 'patio', 'room_diff',
                      'anticipation', 'actual',
                      'long_lat_300', 'trust', 'long_lat_54', 'light', 'lounge',
                      'allowed', 'surprise', 'reduced', 'disgust', 'day', 'garage',
                      'long_lat_225', 'room_sum_desc', 'long_lat_165',
                      'private_laundry', 'swimming', 'long_lat_16', 'long_lat_89',
                      'deck', 'long_lat_324', 'long_lat_12', 'long_lat_35', 'pre',
                      'construction', 'long_lat_338', 'outdoor', 'private_outdoor',
                      'residents', 'long_lat_355', 'balcony', 'storage',
                      'negative', 'live', 'pool', 'public_outdoor', 'bedrooms',
                      'view', 'simplex', 'room_sum',
                      'long_lat_372', 'unit', 'long_lat_320', 'bike', 'long_lat_24',
                      'garden', 'fear', 'speed', 'month', 'cats', 'long_lat_71',
                      'dining', 'sadness', 'anger', 'long_lat_245', 'long_lat_19',
                      'approval', 'joy', 'dishwasher', 'hardwood',
                      'long_lat_21', 'doorman', 'long_lat_303', 'long_lat_145',
                      'high_ceilings', 'war', 'long_lat_67', 'long_lat_68', 'site',
                      'long_lat_359', 'new', 'washer', 'green', 'long_lat_304',
                      'furnished', 'wheelchair', 'long_lat_14',
                      'fitness', 'long_lat_357', 'kitchen', 'high',
                      'appliances', 'long_lat_13', 'long_lat_205', 'building',
                      'long_lat_370', 'long_lat_127', 'prewar', 'concierge',
                      'long_lat_356', 'long_lat_264', 'dogs', 'room_price', 'ceiling',
                      'long_lat_281', 'service', 'parking', 'multi', 'short_term',
                      'long_lat_349', 'laundry', 'closet', 'description_len', 'long_lat_2',
                      'fireplace', 'long_lat_332', 'long_lat_302',
                      'long_lat_20', 'children', 'playroom', 'room_diff_desc', 'private',
                      'long_lat_345', 'long_lat_283',
                      'long_lat_106', 'bathrooms', 'common', 'steel',
                      'positive', 'floors', 'elevator',
                      'long_lat_330', 'roof', 'terrace', 'long_lat_335', 'gym']
BAD_BEHAVING_COLUMNS = ['manager_count', 'listing_id']


def input_fn(df):
    df[LINEAR_CAT_COLUMNS] = df[LINEAR_CAT_COLUMNS].fillna('')
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])
                     for k in DNN_CONT_COLUMNS}
    
    #linear_cont_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])
    #                 for k in LINEAR_CONT_COLUMNS}
    
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
                                           indices=[[i, 0] for i in range(df[k].size)],
                                           values=df[k].values,
                                           dense_shape=[df[k].size, 1])
                        for k in LINEAR_CAT_COLUMNS}
    
    # Merges the two dictionaries into one.
    feature_cols = continuous_cols
    feature_cols.update(categorical_cols)
    #feature_cols.update(linear_cont_cols)
    
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label

def main():
    df_train = pandas.read_csv(TRAIN_DF)
    train_rows = pd.read_csv(os.getcwd() + '/input/train0.95.csv', header=None)[0]
    eval_rows = pd.read_csv(os.getcwd() + '/input/eval0.05.csv', header=None)[0]
    train = df_train.loc[train_rows]
    evaluate = df_train.loc[eval_rows]
    
    # global CONTINUOUS_COLUMNS
    # CONTINUOUS_COLUMNS = list(set(df_train.columns.values) - set(CATEGORICAL_COLUMNS))
    
    deep_columns = list()
    wide_columns = list()
    for col in DNN_CONT_COLUMNS:
        deep_columns.append(tf.contrib.layers.real_valued_column(col))
        # wide_columns.append(tf.contrib.layers.real_valued_column(col))

    # for col in LINEAR_CONT_COLUMNS:
    #    wide_columns.append(tf.contrib.layers.real_valued_column(col))
    
    for col in LINEAR_CAT_COLUMNS:
        col_tensor = tf.contrib.layers.sparse_column_with_hash_bucket(col, hash_bucket_size=1000)
        wide_columns.append(col_tensor)
    
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
                            input_fn=lambda: input_fn(evaluate),
                            eval_steps=1,
                            every_n_steps=50,
                            early_stopping_metric="loss",
                            early_stopping_metric_minimize=True,
                            early_stopping_rounds=400)
     
    # model_dir = tempfile.mkdtemp()
    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir="wide_deep_180col_50hu_dnn_Adagrad",
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[50],
        dnn_optimizer="Adagrad",
        n_classes=3,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
    
    classifier.fit(input_fn=lambda: input_fn(train), 
                   steps=15000, 
                   monitors=[validation_monitor])
    print (validation_monitor)
    
    df_test = pandas.read_csv(TEST_DF)
    df_test[LABEL_COLUMN] = 0
    test_pred_prob = classifier.predict_proba(input_fn=lambda: input_fn(df_test))
    out_df = pd.DataFrame([list(p) for p in test_pred_prob])
    out_df.columns = ["low", "medium", "high"]
    out_df["listing_id"] = df_test.listing_id.values
    out_df.to_csv("wide_deep_nn.csv", index=False)

if __name__ == "__main__":
    os.chdir('/Users/zoya/projects/datascience/tensorflow/tensorflow/com/zoya/tf/renthop/')
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
