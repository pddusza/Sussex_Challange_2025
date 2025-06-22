from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.keras import regularizers


class ModelProviderNoAntiOverfitting:
  def __init__(self, seq_len = 500, nfeatures = 9, nclasses = 8):
    '''
    | Initializer model provider
    | Args:
    | seq_len - length of time series analysed
    | nfeatures - number of features of each time point
    | nclasses - number of target classes
    | Example usage:
    | provider = ModelProvider(500, 9, 8)
    | model = provider.get_model()
    '''
    self.seq_len = seq_len
    self.nfeatures = nfeatures
    self.nclasses = nclasses
    self.model = None


  def build_model(self):
    '''
    | Build model for SHL Challenge
    | Reimplementation of a model from https://github.com/titu1994/MLSTM-FCN for newer keras
    '''

    def squeeze_excite_block(input):
      ''' Create a squeeze-excite block
      Args:
          input: input tensor
          filters: number of output filters
          k: width factor

      Returns: a keras tensor
      '''
      filters = input.shape[-1] # channel_axis = -1 for TF

      se = GlobalAveragePooling1D()(input)
      se = Reshape((1, filters))(se)
      se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
      se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
      se = multiply([input, se])
      return se

    ip = Input(shape=(self.seq_len, self.nfeatures), dtype=float)
    x = Permute((2, 1))(ip)
    x = Masking()(ip)
    x = LSTM(8)(x)

    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform', data_format='channels_last')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform', data_format='channels_last')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform', data_format='channels_last')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(self.nclasses, activation='softmax')(x)
    self.model = Model(ip, out)

  def from_weights(self, weightpath):
    '''
    Get model from existing weights
    '''
    self.build_model()
    self.model.load_weights(weightpath)
    return self.model

  def get_model(self):
    if self.model is None:
      self.build_model()
    return self.model
  


def example_usage(path):
    nao = ModelProviderNoAntiOverfitting(nfeatures=9)
    model = nao.from_weights(path)
    return model


import numpy as np
import pandas as pd
from keras.utils import to_categorical

class SHLDataProviderRaw:
  def __init__(self, root_path:str, flag: str, mode: str, location: str = "Hips", sensor: str = "Acc_x", silent = True, four_locs = False):
    '''
    | Setup data provider for SHL 2025 Challenge
    | Args:
    | root_path: path to root data folder
    | flag: one of train, validation, test
    | mode: one of singlefile, singlefolder, all
    | location: where to load from (Hips/Bag etc)
    | sensor: which file to load (Acc_x/Mag_z etc)
    | silent: supress print statements
    | Example usage:
    | provider = SHLDataProviderRaw(datapath, "train", "singlefolder", location="Bag")
    | x, y = provider.load_data()
    '''
    self.mode = mode.lower()
    self.flag = flag.lower()
    self.location = location.capitalize()
    self.sensor = sensor.capitalize()
    self.silent = silent
    if self.flag not in ["test", "train", "validation"]:
      raise ValueError(f"Unexpected flag received: got {self.flag}")
    if self.mode not in ["singlefile", "singlefolder", "all"]:
      raise ValueError(f"Unexpected mode received: got {self.mode}")
    if self.flag == 'train' and four_locs:
      self.datapath = os.path.join(root_path, "train_4_locations")
    else:
      self.datapath = os.path.join(root_path, self.flag)

  def load_data(self):
    if self.mode == "singlefile":
      x_data, y_data = self._load_from_file()
    elif self.mode == "singlefolder":
      x_data, y_data = self._load_from_folder()
    elif self.mode == "all":
      x_data, y_data = self._load_all()
    return np.array(x_data), np.array(y_data)

  def _load_label_file(self, labelpath):
      if not self.silent:
        print(f"Loading labels from: {labelpath}")
      y_data = np.loadtxt(labelpath, dtype=int)
      y_data[np.isnan(y_data)] = 1
      y_data = np.median(y_data, axis=1).astype(int)
      y_data = y_data - 1
      y_data = to_categorical(y_data, len(np.unique(y_data)))
      return y_data

  def _load_location_data(self, path):
      def load_txt_csv(path):
            if not self.silent:
                print(f"Loading data from: {path}")
            if self.flag == 'test':
                np_data = np.loadtxt(path, dtype=np.float32, delimiter=",")
            else:
                df = pd.read_csv(path, header=None, delim_whitespace=True, engine='python')
                np_data = df.to_numpy()
            return np_data
      acc_x = load_txt_csv(f'{path}/Acc_x.txt')
      acc_y = load_txt_csv(f'{path}/Acc_y.txt')
      acc_z = load_txt_csv(f'{path}/Acc_z.txt')
      gyr_x = load_txt_csv(f'{path}/Gyr_x.txt')
      gyr_y = load_txt_csv(f'{path}/Gyr_y.txt')
      gyr_z = load_txt_csv(f'{path}/Gyr_z.txt')
      mag_x = load_txt_csv(f'{path}/Mag_x.txt')
      mag_y = load_txt_csv(f'{path}/Mag_y.txt')
      mag_z = load_txt_csv(f'{path}/Mag_z.txt')
      data = np.stack([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z], axis=2)
      return data
  
  def _load_from_folder(self):
      '''
      Load all sensors from folder (location)
      '''
      if self.flag == "test":
          filepath = self.datapath
          data = self._load_location_data(filepath)
          return data
      else:
          filepath = os.path.join(self.datapath, self.location)
          labels = self._load_label_file(os.path.join(filepath, "Label.txt"))
          data = self._load_location_data(filepath)
          return data, labels
  def _load_all(self):
      if self.flag == 'test':
          data = self._load_from_folder()
          return data
      self.location = "Hips"
      hips_data, hips_labels = self._load_from_folder()
      self.location = "Torso"
      torso_data, torso_labels = self._load_from_folder()
      self.location = "Bag"
      bag_data, bag_labels = self._load_from_folder()
      self.location = "Hand"
      hand_data, hand_labels = self._load_from_folder()
      data = np.concatenate([hips_data, torso_data, bag_data, hand_data], axis = 0)
      labels = np.concatenate([hips_labels, torso_labels, bag_labels, hand_labels], axis=0)
      return data, labels


def example_usage_data(path):
    dprovider = SHLDataProviderRaw(path, "train", "all", silent=False)
    x, y = dprovider.load_data()
    return x, y