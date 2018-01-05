import keras
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, Flatten, Conv1D, BatchNormalization, MaxPooling1D, Reshape
from config import config as cfg
from keras import regularizers as reg
from dlframework.net_factory.network import Network
import attr
import pdb
import keras.backend.tensorflow_backend as K


@attr.s
class audio_feature_extractor(Network):
    num_classes = attr.ib()
    input_shape = attr.ib()
    transfer = attr.ib(default=False)
    
    def build_model(self, mode=False):
        with tf.device('/cpu:0'):
            PADDING = 'valid'
            print "Input shape prior to processing via 1st conv layer: ", self.input_shape
            model=Sequential()
            model.add(Conv1D(20, 80, activation='relu', input_shape=self.input_shape))
            model.add(MaxPooling1D(8))
            model.add(Conv1D(32, 40, strides=2, activation='relu'))
            model.add(MaxPooling1D(8))
            model.add(Conv1D(64, 32, strides=1, activation='relu'))
            model.add(MaxPooling1D(8))
            model.add(Conv1D(128, 16, strides=2, activation='relu'))
            model.add(MaxPooling1D(4))
            model.add(Conv1D(256, 4, strides=1, activation='relu'))
            model.add(Dropout(.5))

            model.add(Activation('softmax', name = 'audio_feature_extractor'))


            print "Model Summary:"
            print model.summary()    
            return model