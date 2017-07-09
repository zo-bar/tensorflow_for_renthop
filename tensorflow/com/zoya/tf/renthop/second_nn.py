'''
Created on Apr 6, 2017

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
CONTINUOUS_COLUMNS = ['price_ratio', 'bed_price', 'price', 'bed_price_desc', 'feature_count', 
                      'photo_count', 'weekday', 'fee', 'hour',
                      'bed_ratio_desc', 'bedrooms_desc', 'walk', 'ok', 'renovated', 'super', 'latitude', 
                      'level', 'internet', 'mean_bed_price', 'pets', 
                      'exclusive', 
                      'public_laundry', 'luxury', 'central', 
                      'room_price_desc', 'access', 'newly', 'loft', 'granite', 'space', 'old', 
                      'patio', 'room_diff', 'anticipation', 'actual',  
                      'trust', 'light', 'lounge', 
                      'allowed', 'surprise', 'reduced', 'disgust', 'day', 'garage', 
                      'room_sum_desc', 'private_laundry', 'swimming', 
                      'deck', 'pre', 'construction', 'outdoor', 'private_outdoor',
                    'residents', 'balcony', 'storage', 
                    'negative', 'live', 'pool', 'public_outdoor', 'bedrooms', 
                    'view', 'simplex', 'room_sum',
                    'unit', 'bike', 'garden', 'fear', 'speed', 'month', 'cats', 
                    'dining', 'sadness', 'anger', 'approval', 'joy', 'dishwasher', 'location', 
                    'hardwood', 'doorman', 'high_ceilings', 'war', 'site', 'new', 'washer', 
                    'green', 'furnished', 'longitude', 'wheelchair', 
                    'fitness', 'bed_ratio', 'kitchen', 'high', 'appliances', 'building', 
                    'prewar', 'concierge', 'dogs', 'room_price', 'ceiling', 
                    'service', 'parking', 'multi', 'short_term', 'laundry', 'closet', 
                    'description_len', 'fireplace', 'children', 'playroom', 'room_diff_desc', 
                    'private', 'bathrooms', 'common', 'steel', 'positive', 'floors', 
                    'elevator', 'roof', 'terrace', 'gym']

def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])
                     for k in CONTINUOUS_COLUMNS}
    
    # Merges the two dictionaries into one.
    feature_cols = continuous_cols
    
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label

def main():
    df_train = pandas.read_csv(TRAIN_DF)
    df_test = pandas.read_csv(TEST_DF)
    df_test[LABEL_COLUMN] = 0
    
    feature_columns = list()
    for col in CONTINUOUS_COLUMNS:
        feature_columns.append(tf.contrib.layers.real_valued_column(col))
    
    #model_dir = tempfile.mkdtemp()
    classifier = tf.contrib.learn.DNNClassifier(
        model_dir="second_nn_ftrl_100",
        feature_columns=feature_columns,
        hidden_units=[50],
        n_classes=3,
        optimizer="Ftrl") #Ftrl=74, SGD=788, Momentum, RMSProp=78, Adam=78, Adagrad=78
    # best=0.665, Ftrl, 15000 steps, all cont columns, 50hu
        
    classifier.fit(input_fn=lambda: input_fn(df_train), steps=15000)
    
    test_pred_prob = classifier.predict_proba(input_fn=lambda: input_fn(df_test))
    out_df = pd.DataFrame([list(p) for p in test_pred_prob])
    out_df.columns = ["low", "medium", "high"]
    out_df["listing_id"] = df_test.listing_id.values
    out_df.to_csv("second.csv", index=False)

if __name__ == "__main__":
    os.chdir('/Users/zoya/projects/datascience/tensorflow/tensorflow/com/zoya/tf/renthop/')
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
