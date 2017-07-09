'''
Created on Apr 12, 2017

@author: zoya
'''
import os
import pandas
import tensorflow as tf
import pandas as pd

TRAIN_DF = 'train_184.csv'
TEST_DF = 'plain_test_df.csv'

LABEL_COLUMN = "out"
LINEAR_CAT_COLUMNS = ['building_id', 'manager_id', 'street_address', 'display_address']
DNN_CONT_COLUMNS = ["bathrooms", "bedrooms", "latitude", "longitude",
                    "price", "location", "month", "day", "weekday", "hour", "feature_count", 
                    "photo_count", "description_len", "bed_price" ]
#"has_photos", "has_features"

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

    deep_columns = list()
    wide_columns = list()
    for col in DNN_CONT_COLUMNS:
        deep_columns.append(tf.contrib.layers.real_valued_column(col))
    
    for col in LINEAR_CAT_COLUMNS:
        wide_columns.append(tf.contrib.layers.sparse_column_with_hash_bucket(col, hash_bucket_size=1000))
    
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
                            input_fn=lambda: input_fn(evaluate),
                            eval_steps=1,
                            every_n_steps=100,
                            early_stopping_metric="loss",
                            early_stopping_metric_minimize=True,
                            early_stopping_rounds=400)
     
    # model_dir = tempfile.mkdtemp()
    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir="wide_deep_22col_10hu_dnn_Ftrl",
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[10],
        dnn_optimizer="Ftrl",
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