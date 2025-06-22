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