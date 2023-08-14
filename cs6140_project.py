import numpy as np
import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.run_functions_eagerly(True )

feature_list = []
fire_map_info = {'name':'FIRMS', 'band' :'T21', 'min': 300, 'max': 509.29}
veg_map_info = {'name':'MODIS/MOD09GA_006_NDVI', 'band' :'NDVI', 'min': -1, 'max': 1}
feature_list.append(veg_map_info)
leaf_map_info = {'name':'MODIS/061/MCD15A3H', 'band' :'Lai', 'min': 0, 'max': 30}
feature_list.append(leaf_map_info)
soil_moist_map_info = {'name':'NASA/FLDAS/NOAH01/C/GL/M/V001', 'band' :'SoilMoi00_10cm_tavg', 'min': 0, 'max': 1}
feature_list.append(soil_moist_map_info)
temp_map_info = {'name':'MODIS/061/MOD11A1', 'band' :'LST_Day_1km', 'min': 13658, 'max': 15658}
feature_list.append(temp_map_info)
precipitation_map_info = {'name':'ECMWF/ERA5/MONTHLY', 'band' :'total_precipitation', 'min': 0, 'max': 0.4}
feature_list.append(precipitation_map_info)
wind_speed_u_map_info = {'name':'ECMWF/ERA5/MONTHLY', 'band' :'u_component_of_wind_10m', 'min': -8.7, 'max': 8.7}
feature_list.append(wind_speed_u_map_info)
wind_speed_v_map_info = {'name':'ECMWF/ERA5/MONTHLY', 'band' :'v_component_of_wind_10m', 'min': -6.8, 'max': 6.8}
feature_list.append(wind_speed_v_map_info)

LABEL = fire_map_info['band']
BANDS = []
for feature in feature_list:
  BANDS += [feature['band']]
FEATURE_NAMES = BANDS + [LABEL]

PROJECT = 'ee-my-char'
DATA_BUCKET = '6140-data-bucket'
OUTPUT_BUCKET = '6140-output-bucket'
# Specify names locations for outputs in Cloud Storage.
FOLDER = 'fcnn-demo'
MODEL_FOLDER = 'model_output'
TRAINING_BASE = 'training_patches'
EVAL_BASE = 'eval_patches'
# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = 256
# KERNEL_SIZE = 128
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
# List of fixed-length features, all of which are float32.
columns = [tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURE_NAMES]
FEATURES_DICT = dict(zip(FEATURE_NAMES, columns))

# Specify model training parameters.
TRAIN_SIZE = 16000
EVAL_SIZE = 8000
BATCH_SIZE = 64
EPOCHS = 10
BUFFER_SIZE = 2000
OPTIMIZER = 'adam'
LOSS = 'MeanSquaredError'
METRICS = ['RootMeanSquaredError']
TRAINING_PATH = 'gs://' + DATA_BUCKET + '/' + FOLDER + '/' + 'training*'
EVAL_PATH = 'gs://' + DATA_BUCKET + '/' + FOLDER + '/' + 'eval*'
MODEL_SAVE_PATH = 'gs://' + DATA_BUCKET + '/' + MODEL_FOLDER


def parse_tfrecord(example_proto):
  """The parsing function.
  Read a serialized example into the structure defined by FEATURES_DICT.
  Args:
    example_proto: a serialized Example.
  Returns:
    A dictionary of tensors, keyed by feature name.
  """
  # print(f'Example Proto: {example_proto}')
  parsed_features = tf.io.parse_single_example(example_proto, FEATURES_DICT)
  print(f'Parsed features: {parsed_features}')
  print()
  return parsed_features

def to_tuple(inputs, deb=True):
  """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
  Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
  Args:
    inputs: A dictionary of tensors, keyed by feature name.
  Returns:
    A tuple of (inputs, outputs).
  """
  inputsList = [inputs.get(key) for key in FEATURE_NAMES]
  stacked = tf.stack(inputsList, axis=0)

  if deb:
    print(f'to_tuple inputsList {inputsList}')
    print(f'to_tuple len list {len(inputsList)}')
    print(f'to_tuple stacked shape {stacked.shape}')

  # Convert from CHW to HWC
  # See https://caffe2.ai/docs/tutorial-image-pre-processing.html
  stacked = tf.transpose(stacked, [1, 2, 0])

  if deb:
    print(f'to_tuple after reshape {stacked.shape}')
    print(f'length of BANDS??? {len(BANDS)}')
  return stacked[:,:,:len(BANDS)], stacked[:,:,len(BANDS):]


def get_dataset(pattern):
  """Function to read, parse and format to tuple a set of input tfrecord files.
  Get all the files matching the pattern, parse and convert to tuple.
  Args:
    pattern: A file pattern to match in a Cloud Storage bucket.
  Returns:
    A tf.data.Dataset
  """
  glob = tf.io.gfile.glob(pattern)
  dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
  print(f'get_dataset:parse step1 {dataset}')
  dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
  print(f'get_dataset step1-2 {dataset.element_spec}')
  dataset = dataset.map(to_tuple, num_parallel_calls=5)
  return dataset


def get_training_dataset(filename_pattern):
	"""Get the preprocessed training dataset
  Returns:
    A tf.data.Dataset of training data.
  """
	glob = filename_pattern
	dataset = get_dataset(glob)
	dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
	return dataset

def get_eval_dataset(filename_pattern):
	"""Get the preprocessed training dataset
  Returns:
    A tf.data.Dataset of training data.
  """
	glob = filename_pattern
	dataset = get_dataset(glob)
	dataset = dataset.batch(1).repeat()
	return dataset

training = get_training_dataset(TRAINING_PATH)
evaluation = get_eval_dataset(EVAL_PATH)

import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
import tensorflow.keras.metrics as metrics
import tensorflow.keras.optimizers as optimizers

def conv_block(input_tensor, num_filters):
	encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
	encoder = layers.BatchNormalization()(encoder)
	encoder = layers.Activation('relu')(encoder)
	encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
	encoder = layers.BatchNormalization()(encoder)
	encoder = layers.Activation('relu')(encoder)
	return encoder

def encoder_block(input_tensor, num_filters):
	encoder = conv_block(input_tensor, num_filters)
	encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
	return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
	decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
	decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
	decoder = layers.BatchNormalization()(decoder)
	decoder = layers.Activation('relu')(decoder)
	decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
	decoder = layers.BatchNormalization()(decoder)
	decoder = layers.Activation('relu')(decoder)
	decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
	decoder = layers.BatchNormalization()(decoder)
	decoder = layers.Activation('relu')(decoder)
	return decoder

def get_model():
	inputs = layers.Input(shape=[None, None, len(BANDS)]) # 256
	encoder0_pool, encoder0 = encoder_block(inputs, 32) # 128
	encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
	encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
	encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
	encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
	center = conv_block(encoder4_pool, 1024) # center
	decoder4 = decoder_block(center, encoder4, 512) # 16
	decoder3 = decoder_block(decoder4, encoder3, 256) # 32
	decoder2 = decoder_block(decoder3, encoder2, 128) # 64
	decoder1 = decoder_block(decoder2, encoder1, 64) # 128
	decoder0 = decoder_block(decoder1, encoder0, 32) # 256
	outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

	model = models.Model(inputs=[inputs], outputs=[outputs])

	model.compile(
		optimizer=optimizers.get(OPTIMIZER),
		loss=losses.get(LOSS),
		metrics=[metrics.get(metric) for metric in METRICS],
		run_eagerly=True
		)

	return model


from tqdm import tqdm

class TQDMCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super(TQDMCallback, self).__init__()
        self.total_epochs = total_epochs
        self.pbar = tqdm(total=self.total_epochs, unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        self.pbar.set_description(f"Epoch {epoch + 1}/{self.total_epochs}, Loss: {logs['loss']:.4f}, "
                                  f"RMSE: {logs['root_mean_squared_error']:.4f}, ")
                                  # f"Val_Loss: {logs['val_loss']:.4f}, "
                                  # f"Val_RMSE: {logs['val_root_mean_squared_error']:.4f}")

    def on_train_end(self, logs=None):
        self.pbar.close()


tqdm_callback = TQDMCallback(total_epochs=EPOCHS)

m = get_model()

m.summary()

m.fit(
    x=training,
    epochs=10,
    steps_per_epoch=int(16000 / 16),
    validation_data=evaluation,
    validation_steps=8000,
    callbacks=[tqdm_callback],
)

m.save(MODEL_SAVE_PATH)