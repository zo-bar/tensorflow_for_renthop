'''
Created on Apr 5, 2017

@author: zoya
'''
import os
import pandas
import tensorflow as tf
import pandas as pd
import tempfile

TRAIN_DF = 'train_184.csv'
TEST_DF = 'test_183.csv'

LABEL_COLUMN = "out"
CATEGORICAL_COLUMNS = ['building_id', 'manager_id', 'street_address', 
                       'display_address', 'display_address_cleaned']
CONTINUOUS_COLUMNS = ['long_lat_185', 'walk', 'ok', 'long_lat_339', 
                      'long_lat_126', 'renovated', 'super', 'latitude', 
                      'level', 'price', 'internet', 'mean_bed_price', 'pets', 
                      'feature_count', 'exclusive', 'long_lat_336', 
                      'public_laundry', 'luxury', 'long_lat_362', 'central', 
                      'room_price_desc', 'long_lat_52', 'access', 'newly', 
                      'long_lat_333', 'long_lat_317', 'loft', 'long_lat_11', 
                      'long_lat_107', 'granite', 'space', 'old', 'long_lat_6', 
                      'long_lat_373', 'long_lat_105', 'patio', 'room_diff', 
                      'anticipation', 'actual', 'weekday', 'fee', 'price_ratio', 
                      'long_lat_300', 'trust', 'long_lat_54', 'light', 'lounge', 
                      'allowed', 'surprise', 'reduced', 'disgust', 'day', 'garage', 
                      'long_lat_225', 'room_sum_desc', 'long_lat_165', 'bed_price_desc', 
                      'private_laundry', 'swimming', 'long_lat_16', 'long_lat_89', 
                      'deck', 'long_lat_324', 'long_lat_12', 'long_lat_35', 'pre', 
                      'construction', 'long_lat_338', 'outdoor', 'private_outdoor', 
                      'residents', 'photo_count', 'long_lat_355', 'balcony', 'storage', 
                      'negative', 'hour', 'live', 'pool', 'public_outdoor', 'bedrooms', 
                      'bed_ratio_desc', 'bedrooms_desc', 'view', 'simplex', 'room_sum',
                      'long_lat_372', 'unit', 'long_lat_320', 'bike', 'long_lat_24', 
                      'garden', 'fear', 'speed', 'month', 'cats', 'long_lat_71', 
                      'dining', 'sadness', 'anger', 'long_lat_245', 'long_lat_19', 
                      'approval', 'joy', 'dishwasher', 'location', 'hardwood', 
                      'long_lat_21', 'doorman', 'long_lat_303', 'long_lat_145', 
                      'high_ceilings', 'war', 'long_lat_67', 'long_lat_68', 'site', 
                      'long_lat_359', 'new', 'washer', 'green', 'long_lat_304', 
                      'furnished', 'longitude', 'bed_price', 'wheelchair', 'long_lat_14', 
                      'fitness', 'bed_ratio', 'long_lat_357', 'kitchen', 'high', 
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
DNN_COLUMNS = ['manager_count', 'listing_id']

def input_fn(df):
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].fillna('')
    
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])
                     for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
                                           indices=[[i, 0] for i in range(df[k].size)],
                                           values=df[k].values,
                                           dense_shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
    
    # Merges the two dictionaries into one.
    feature_cols = continuous_cols
    feature_cols.update(categorical_cols)
    
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label

def main():
    df_train = pandas.read_csv(TRAIN_DF)
    df_test = pandas.read_csv(TEST_DF)
    df_test[LABEL_COLUMN] = 0
    train_rows = pd.read_csv(os.getcwd() +'/input/train0.95.csv', header=None)[0]
    test_rows = pd.read_csv(os.getcwd() +'/input/eval0.05.csv', header=None)[0]
    train = df_train.loc[train_rows]
    evaluate = df_train.loc[test_rows]
    #global CONTINUOUS_COLUMNS
    #CONTINUOUS_COLUMNS = list(set(df_train.columns.values) - set(CATEGORICAL_COLUMNS))
    
    deep_columns = list()
    wide_columns = list()
    for c in CONTINUOUS_COLUMNS:
        deep_columns.append(tf.contrib.layers.real_valued_column(c))
        #wide_columns.append(tf.contrib.layers.real_valued_column(c))
    
    for col in CATEGORICAL_COLUMNS:
        col_tensor = tf.contrib.layers.sparse_column_with_hash_bucket(col, hash_bucket_size=100)
        wide_columns.append(col_tensor)
    
    #model_dir = tempfile.mkdtemp()
    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir="wide_deep_180col_50hu_ftrl",
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[50],
        dnn_optimizer="Ftrl", 
        n_classes=3)
        
    classifier.fit(input_fn=lambda: input_fn(df_train), steps=15000)
    
        # Fit model.
    for i in range(1, 15):
        classifier.fit(input_fn=lambda: input_fn(train), steps=1000)
        ev = classifier.evaluate(input_fn=lambda: input_fn(evaluate), steps=1)
        print("Evaluated: " + str(i) + ": "+ str(ev))
        with open(os.getcwd() + '/wide_deep_eval.txt', 'a+') as thefile:
            thefile.write("\nEval:" + str(ev) + "   Wide_deep " + str(len(wide_columns)+ len(deep_columns)) + " cols, steps:" + str(i * 1000) + ", hu=50, Ftrl")


    test_pred_prob = classifier.predict_proba(input_fn=lambda: input_fn(df_test))
    out_df = pd.DataFrame([list(p) for p in test_pred_prob])
    out_df.columns = ["low", "medium", "high"]
    out_df["listing_id"] = df_test.listing_id.values
    out_df.to_csv("wide_deep_nn.csv", index=False)

if __name__ == "__main__":
    os.chdir('/Users/zoya/projects/datascience/tensorflow/tensorflow/com/zoya/tf/renthop/')
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
