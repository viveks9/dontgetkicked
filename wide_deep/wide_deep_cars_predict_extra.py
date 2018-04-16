# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.arg_parsers import parsers
from official.utils.logs import hooks_helper
from official.utils.misc import model_helpers

_CSV_COLUMNS = [
    'ref_id','is_bad_buy', 'purch_date', 'auction', 'veh_year', 
    'vehicle_age', 'vehicle_make', 'vehicle_model', 'trim', 'sub_model', 
    'color', 'transmission', 'wheel_type_id', 'wheel_type', 'vehicle_odometer', 
    'nationality', 'size', 'top_three_american_name', 'mmr_acquisition_auction_average_price', 'mmr_acquisition_auction_clean_price', 
    'mmr_acquisition_retail_average_price', 'mmr_acquisiton_retail_clean_price', 'mmr_current_auction_average_price', 'mmr_current_auction_clean_price', 'mmr_current_retail_average_price', 
    'mmr_current_retail_clean_price', 'prime_unit', 'aucguart', 'byr_no', 'vin_zip', 
    'vin_state', 'vehb_cost', 'is_online_sale', 'warranty_cost', 'price_comparison_auction_retail',
    'ratio_odometer_age', 'purchase_month'
]

_CSV_COLUMN_DEFAULTS = [[0], [0], [''], [''], [''], 
                        [0], [''], [''], [''], [''],
                        [''], [''], [''], [''], [0],
                        [''], [''], [''], [0], [0],
                        [0], [0], [0], [0], [0],
                        [0], [''], [''], [0], [''],
                        [''], [0], [0], [0], ['NEQ'], [0], [0]]
            
_NUM_EXAMPLES = {
    'train': 101471,
    'validation': 8096,
}


LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def build_model_columns():
  """Builds a set of wide and deep feature columns."""

  # Continuous columns
  vehicle_age = tf.feature_column.numeric_column('vehicle_age')
  vehicle_odometer = tf.feature_column.numeric_column('vehicle_odometer')

  mmr_acquisition_auction_average_price = tf.feature_column.numeric_column('mmr_acquisition_auction_average_price')
  mmr_acquisition_auction_clean_price = tf.feature_column.numeric_column('mmr_acquisition_auction_clean_price')
  mmr_acquisition_retail_average_price = tf.feature_column.numeric_column('mmr_acquisition_retail_average_price')
  mmr_acquisiton_retail_clean_price = tf.feature_column.numeric_column('mmr_acquisiton_retail_clean_price')


  mmr_current_auction_average_price = tf.feature_column.numeric_column('mmr_current_auction_average_price')
  mmr_current_auction_clean_price = tf.feature_column.numeric_column('mmr_current_auction_clean_price')
  mmr_current_retail_average_price = tf.feature_column.numeric_column('mmr_current_retail_average_price')
  mmr_current_retail_clean_price = tf.feature_column.numeric_column('mmr_current_retail_clean_price')


  vehb_cost = tf.feature_column.numeric_column('vehb_cost')
  warranty_cost = tf.feature_column.numeric_column('warranty_cost')

  is_online_sale = tf.feature_column.numeric_column('is_online_sale')
  buyer_no = tf.feature_column.numeric_column('byr_no')
  ratio_odometer_age = tf.feature_column.numeric_column('ratio_odometer_age')
  purchase_month = tf.feature_column.numeric_column('purchase_month')
  #categorical columns

  # To show an example of hashing:
  auction = tf.feature_column.categorical_column_with_hash_bucket(
      'auction', hash_bucket_size=1000)
  vehicle_make = tf.feature_column.categorical_column_with_hash_bucket(
      'vehicle_make', hash_bucket_size=1000)
  vehicle_model = tf.feature_column.categorical_column_with_hash_bucket(
      'vehicle_model', hash_bucket_size=1000)
  trim = tf.feature_column.categorical_column_with_hash_bucket(
      'trim', hash_bucket_size=1000)
  sub_model = tf.feature_column.categorical_column_with_hash_bucket(
      'sub_model', hash_bucket_size=1000)
  color = tf.feature_column.categorical_column_with_hash_bucket(
      'color', hash_bucket_size=1000)
  transmission = tf.feature_column.categorical_column_with_hash_bucket(
      'transmission', hash_bucket_size=1000)
  wheel_type = tf.feature_column.categorical_column_with_hash_bucket(
      'wheel_type', hash_bucket_size=1000)
  nationality = tf.feature_column.categorical_column_with_hash_bucket(
      'nationality', hash_bucket_size=1000)
  vehicle_size = tf.feature_column.categorical_column_with_hash_bucket(
      'size', hash_bucket_size=1000)
  top_three_american_name = tf.feature_column.categorical_column_with_hash_bucket(
      'top_three_american_name', hash_bucket_size=1000)
  #buyer_no = tf.feature_column.categorical_column_with_hash_bucket(
  #    'byr_no', hash_bucket_size=1000)
  vin_zip = tf.feature_column.categorical_column_with_hash_bucket(
      'vin_zip', hash_bucket_size=1000)
  vin_state = tf.feature_column.categorical_column_with_hash_bucket(
      'vin_state', hash_bucket_size=1000)
  price_comparison_auction_retail = tf.feature_column.categorical_column_with_hash_bucket(
      'price_comparison_auction_retail', hash_bucket_size=1000)
  #is_online_sale = tf.feature_column.categorical_column_with_hash_bucket(
  #    'is_online_sale', hash_bucket_size=1000)
    

  # Transformations.
  age_buckets = tf.feature_column.bucketized_column(
      vehicle_age, boundaries=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
  odo_buckets = tf.feature_column.bucketized_column(
      vehicle_odometer, boundaries=[50000, 75000, 100000, 125000, 150000, 175000])
  mmr_current_auction_avg_price_buckets = tf.feature_column.bucketized_column(
      mmr_current_auction_average_price, 
      boundaries=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 
      24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000])
  mmr_current_retail_avg_price_buckets = tf.feature_column.bucketized_column(
      mmr_current_retail_average_price, 
      boundaries=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 
      24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000])

  # Wide columns and deep columns.
#  base_columns = [
#      age_buckets, odo_buckets, mmr_acquisition_auction_average_price, mmr_acquisition_auction_clean_price, 
#      mmr_acquisition_retail_average_price, mmr_acquisiton_retail_clean_price, mmr_current_auction_average_price,
#      mmr_current_auction_clean_price, mmr_current_retail_average_price, mmr_current_retail_clean_price, vehb_cost,
#      warranty_cost,
#  ]

  base_columns = [
      #age_buckets, odo_buckets, 
      vehicle_age, vehicle_odometer,
      mmr_acquisition_auction_average_price, 
      mmr_acquisition_auction_clean_price, 
      mmr_acquisition_retail_average_price, 
      mmr_acquisiton_retail_clean_price, 
      mmr_current_auction_average_price,
      mmr_current_auction_clean_price, 
      mmr_current_retail_average_price, 
      mmr_current_retail_clean_price, 
      vehb_cost,
      warranty_cost,
      purchase_month,
      ratio_odometer_age,
      #categorical columns
      auction,
      vehicle_make,
      vehicle_model,
      trim,
      sub_model,
      color,
      transmission,
      wheel_type,
      nationality,
      vehicle_size,
      top_three_american_name,
      buyer_no,
      vin_zip,
      vin_state,
      is_online_sale, 
      price_comparison_auction_retail, 
  ]


  crossed_columns = [
      tf.feature_column.crossed_column(
          ['wheel_type', 'auction'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          ['wheel_type', 'vin_zip'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          [mmr_current_auction_avg_price_buckets, mmr_current_retail_avg_price_buckets], hash_bucket_size=1000),
      #tf.feature_column.crossed_column(
      #    [age_buckets, 'vehicle_make', 'vehicle_model'], hash_bucket_size=1000),
      #tf.feature_column.crossed_column(
      #    [odo_buckets, 'vehicle_make', 'vehicle_model'], hash_bucket_size=1000),
  ]

  wide_columns = base_columns + crossed_columns
  #wide_columns = base_columns

  deep_columns = [
      vehicle_age, 
      vehicle_odometer,
      mmr_acquisition_auction_average_price, 
      mmr_acquisition_auction_clean_price, 
      mmr_acquisition_retail_average_price, 
      mmr_acquisiton_retail_clean_price, 
      mmr_current_auction_average_price,
      mmr_current_auction_clean_price, 
      mmr_current_retail_average_price, 
      mmr_current_retail_clean_price, 
      vehb_cost,
      warranty_cost,
      purchase_month,
      ratio_odometer_age,
      buyer_no,
      #categorical columns
      tf.feature_column.indicator_column(auction),
      tf.feature_column.indicator_column(vehicle_make),
      tf.feature_column.indicator_column(vehicle_model),
      tf.feature_column.indicator_column(trim),
      tf.feature_column.indicator_column(sub_model),
      tf.feature_column.indicator_column(color),
      tf.feature_column.indicator_column(transmission),
      tf.feature_column.indicator_column(wheel_type),
      tf.feature_column.indicator_column(nationality),
      tf.feature_column.indicator_column(vehicle_size),
      tf.feature_column.indicator_column(top_three_american_name),
      tf.feature_column.indicator_column(price_comparison_auction_retail),
      tf.feature_column.indicator_column(vin_zip),
      tf.feature_column.indicator_column(vin_state),
      #tf.feature_column.indicator_column(is_online_sale),  
  
      # To show an example of embedding
      # tf.feature_column.embedding_column(occupation, dimension=8),
  ]

  return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        optimizer=tf.train.FtrlOptimizer(
          learning_rate=0.1,
          l1_regularization_strength=1.0,
          l2_regularization_strength=1.0),
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run data_download.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    #columns = tf.decode_csv(value)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('is_bad_buy')
    #return features, tf.equal(labels, '>50K')
    return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset


def main(argv):
  parser = WideDeepArgParser()
  flags = parser.parse_args(args=argv[1:])

  # Clean up the model directory if present
  #shutil.rmtree(flags.model_dir, ignore_errors=True)
  model = build_estimator(flags.model_dir, flags.model_type)

  predict_file = os.path.join(flags.data_dir, 'cars_to_predict.csv')

  def predict_input_fn():
    return input_fn(predict_file, 1, False, flags.batch_size)

  loss_prefix = LOSS_PREFIX.get(flags.model_type, '')
  train_hooks = hooks_helper.get_train_hooks(
      flags.hooks, batch_size=flags.batch_size,
      tensors_to_log={'average_loss': loss_prefix + 'head/truediv',
                      'loss': loss_prefix + 'head/weighted_loss/Sum'})

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  pred_iter = model.predict(input_fn=predict_input_fn)
  for pred in pred_iter:
    class_id = pred['class_ids'][0]
    probability = pred['probabilities'][class_id]
    print(pred['classes'])
    print("class:{0}, probability{1}".format(class_id, probability))


class WideDeepArgParser(argparse.ArgumentParser):
  """Argument parser for running the wide deep model."""

  def __init__(self):
    super(WideDeepArgParser, self).__init__(parents=[
        parsers.BaseParser(multi_gpu=False, num_gpu=False)])
    self.add_argument(
        '--model_type', '-mt', type=str, default='wide',
        choices=['wide', 'deep', 'wide_deep'],
        help='[default %(default)s] Valid model types: wide, deep, wide_deep.',
        metavar='<MT>')
    self.set_defaults(
        data_dir='/home/vivek/Work/kaggle/DontGetKicked/data/train',
        model_dir='/home/vivek/Work/kaggle/DontGetKicked/model',
        train_epochs=400,
        epochs_between_evals=2,
        batch_size=80)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
