    col_tensor = tf.contrib.layers.sparse_column_with_hash_bucket(col, hash_bucket_size=100)
    wide_columns.append(col_tensor)
model_dir = tempfile.mkdtemp()
classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50],
    n_classes=3)
deep_columns
wide_columns
    global CONTINUOUS_COLUMNS = list(set(df_train.columns.values[1:200]) - set(CATEGORICAL_COLUMNS))
CONTINUOUS_COLUMNS = list(set(df_train.columns.values[1:200]) - set(CATEGORICAL_COLUMNS))
CONTINUOUS_COLUMNS
len(CONTINUOUS_COLUMNS)
feature_columns = list()
for c in CONTINUOUS_COLUMNS:
    feature_columns.append(tf.contrib.layers.real_valued_column(c, dimension=3))
for col in CATEGORICAL_COLUMNS:
    col_tensor = tf.contrib.layers.sparse_column_with_hash_bucket(col, hash_bucket_size=100)
    feature_columns.append(col_tensor)
model_dir = tempfile.mkdtemp()
classifier  = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns, 
                                                model_dir=model_dir,
                                                n_classes=3)
classifier.fit(input_fn=lambda: input_fn(df_train), steps=10000)
df_train.columns.values
CONTINUOS_COLUMNS.remove('row')
CONTINUOUS_COLUMNS.remove('row')
col_names = df_train.columns.values
colnames.remove('row')
col_names.remove('row')
col_names = list(df_train.columns.values)
col_names.remove('row')
col_names.remove('out')
col_names
col_names = df_train.columns.values
col_names.remove('row')
col_names.remove('out')
CONTINUOUS_COLUMNS = list(set(col_names) - set(CATEGORICAL_COLUMNS))
col_names = list(df_train.columns.values)
col_names.remove('row')
col_names.remove('out')
CONTINUOUS_COLUMNS = list(set(col_names) - set(CATEGORICAL_COLUMNS))
feature_columns = list()
for c in CONTINUOUS_COLUMNS:
    feature_columns.append(tf.contrib.layers.real_valued_column(c, dimension=3))
for col in CATEGORICAL_COLUMNS:
    col_tensor = tf.contrib.layers.sparse_column_with_hash_bucket(col, hash_bucket_size=100)
    feature_columns.append(col_tensor)
model_dir = tempfile.mkdtemp()
classifier  = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns, 
                                                model_dir=model_dir,
                                                n_classes=3)
# Fit model.
classifier.fit(input_fn=lambda: input_fn(df_train), steps=10000)
len(CONTINUOUS_COLUMNS)
def input_fn(df):
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].fillna('')
    df[CONTINUOUS_COLUMNS] = df[CONTINUOUS_COLUMNS].fillna(0)
    # print("Categorical columns: " + str(CATEGORICAL_COLUMNS))
    # print("Continuous columns: " + str(CONTINUOUS_COLUMNS))
    #     
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
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
classifier  = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns, 
                                                    model_dir=model_dir,
                                                    n_classes=3)
classifier.fit(input_fn=lambda: input_fn(df_train), steps=10000)
model_dir = tempfile.mkdtemp()
classifier.fit(input_fn=lambda: input_fn(df_train), steps=10000)
import sys; print('%s %s' % (sys.executable or sys.platform, sys.version))
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
CONTINUOUS_COLUMNS = ['price', 'price_ratio', 'mean_bed_price']
def input_fn(df):
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].fillna('')
    df[CONTINUOUS_COLUMNS] = df[CONTINUOUS_COLUMNS].fillna(0)
    # print("Categorical columns: " + str(CATEGORICAL_COLUMNS))
    # print("Continuous columns: " + str(CONTINUOUS_COLUMNS))
    #     
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
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
os.chdir('/Users/zoya/projects/datascience/tensorflow/tensorflow/com/zoya/tf/renthop/')
    
tf.logging.set_verbosity(tf.logging.INFO)
df_train = pandas.read_csv(TRAIN_DF)
df_test = pandas.read_csv(TEST_DF)
df_test[LABEL_COLUMN] = 0
    col_names = list(df_train.columns.values)
col_names.remove('row')
col_names.remove('out')
CONTINUOUS_COLUMNS = list(set(col_names) - set(CATEGORICAL_COLUMNS))
col_names = list(df_train.columns.values)
col_names.remove('row')
col_names.remove('out')
CONTINUOUS_COLUMNS = list(set(col_names) - set(CATEGORICAL_COLUMNS))
CONTINUOUS_COLUMNS
CONTINUOUS_COLUMNS = ['long_lat_185', 'walk', 'ok', 'long_lat_339', 
                      'long_lat_126', 'renovated', 'super', 'latitude', 
                      'level', 'price', 'internet', 'mean_bed_price', 'pets', 
                      'feature_count', 'exclusive', 'long_lat_336', 
                      'public_laundry', 'luxury', 'long_lat_362', 'central', 
                      'room_price_desc', 'long_lat_52', 'access', 'newly', 
                      'long_lat_333', 'long_lat_317', 'loft', 'long_lat_11', 
                      'long_lat_107', 'granite', 'space', 'old', 'long_lat_6', 
                      'long_lat_373', 'long_lat_105', 'patio', 'room_diff', 
                      'anticipation', 'actual', 'weekday', 'fee', 'price_ratio']
len(CONTINUOUS_COLUMNS)
feature_columns = list()
for c in CONTINUOUS_COLUMNS:
    feature_columns.append(tf.contrib.layers.real_valued_column(c))
for col in CATEGORICAL_COLUMNS:
    col_tensor = tf.contrib.layers.sparse_column_with_hash_bucket(col, hash_bucket_size=100)
    feature_columns.append(col_tensor)
model_dir = tempfile.mkdtemp()
classifier  = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns, 
                                                    model_dir=model_dir,
                                                    n_classes=3)
classifier.fit(input_fn=lambda: input_fn(df_train), steps=10000)
    test_pred_prob = classifier.predict_proba(input_fn=lambda: input_fn(df_test))
test_pred_prob = classifier.predict_proba(input_fn=lambda: input_fn(df_test))
out_df = pd.DataFrame([list(p) for p in test_pred_prob])
out_df.head(10)
min(out_df[0])
min(out_df[2])
min(out_df[2])
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
                      'long_lat_345', 'children', 'long_lat_283', 'listing_id', 
                      'positive', 'long_lat_106', 'floors', 'elevator', 'playroom', 
                      'long_lat_332', 'bathrooms', 'common', 'steel', 'long_lat_302', 
                      'long_lat_20', 'manager_count', 'room_diff_desc', 'private', 
                      'long_lat_330', 'roof', 'terrace', 'long_lat_335', 'gym', 
                      'long_lat_372', 'unit', 'long_lat_320', 'bike', 'long_lat_24', 
                      'garden', 'fear', 'speed', 'month', 'cats', 'long_lat_71', 
                      'dining', 'sadness', 'anger', 'long_lat_245', 'long_lat_19', 
                      'approval', 'joy', 'dishwasher', 'location', 'hardwood']
feature_columns = list()
for c in CONTINUOUS_COLUMNS:
    feature_columns.append(tf.contrib.layers.real_valued_column(c))
for col in CATEGORICAL_COLUMNS:
    col_tensor = tf.contrib.layers.sparse_column_with_hash_bucket(col, hash_bucket_size=100)
    feature_columns.append(col_tensor)
model_dir = tempfile.mkdtemp()
classifier  = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns, 
                                                    model_dir=model_dir,
                                                    n_classes=3)
