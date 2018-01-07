'''This module contains all of model used'''

from util import Dataset, Preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Convolution1D, Embedding
from metric import precision, recall

SEQUENCE_MAX_LENGTH = 10
NUM_EPOCHS = 30


class Model1:
    '''\
    This model consist of\
    Preprocessing: Noise Removal, Case Folding, Tokenization
    Architecture: Conv1D, ReLU, MaxPooling
    '''

    @staticmethod
    def get_meta():
        '''Return name of this model'''
        return 'one'

    @staticmethod
    def get_description():
        '''Description for visualizer legend'''
        return 'Arch A + NR + CF + TKN'

    @staticmethod
    def get_dataset():
        '''Used for load the dataset and preprocess it'''
        data = Dataset()
        data.remove_noise()
        data.fold_case()
        data.tokenize()
        return data

    @staticmethod
    def preprocess(text):
        '''Used for preprocess the text in production environment'''
        text = Preprocessing.remove_noise(text)
        text = Preprocessing.fold_case(text)
        text = Preprocessing.tokenize(text)
        return text

    @staticmethod
    def get_network(input_dim, total_category):
        '''Used for get the architecture of this model'''
        model = Sequential()
        model.add(Embedding(
            input_dim,
            60,
            input_length=SEQUENCE_MAX_LENGTH))
        model.add(Dropout(0.2))
        model.add(Convolution1D(filters=100,
                                padding='valid',
                                kernel_size=2,
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D())
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(total_category, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', precision, recall])
        return model

class Model2:
    '''\
    This model consist of\
    Preprocessing: Noise Removal, Case Folding, Tokenization
    Architecture: Conv1D, ReLU, MaxPooling
    '''

    @staticmethod
    def get_meta():
        '''Return name of this model'''
        return 'two'

    @staticmethod
    def get_description():
        '''Description for visualizer legend'''
        return 'Arch A + NR + CF + SWR + TKN'

    @staticmethod
    def get_dataset():
        '''Used for load the dataset and preprocess it'''
        data = Dataset()
        data.remove_noise()
        data.fold_case()
        data.remove_stopword()
        data.tokenize()
        return data

    @staticmethod
    def preprocess(text):
        '''Used for preprocess the text in production environment'''
        text = Preprocessing.remove_noise(text)
        text = Preprocessing.fold_case(text)
        text = Preprocessing.remove_stopword(text)
        text = Preprocessing.tokenize(text)
        return text

    @staticmethod
    def get_network(input_dim, total_category):
        '''Used for get the architecture of this model'''
        model = Sequential()
        model.add(Embedding(
            input_dim,
            60,
            input_length=SEQUENCE_MAX_LENGTH))
        model.add(Dropout(0.2))
        model.add(Convolution1D(filters=100,
                                padding='valid',
                                kernel_size=2,
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D())
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(total_category, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', precision, recall])
        return model

class Model3:
    '''\
    This model consist of\
    Preprocessing: Noise Removal, Case Folding, Tokenization
    Architecture: Conv1D, ReLU, MaxPooling
    '''

    @staticmethod
    def get_meta():
        '''Return name of this model'''
        return 'three'

    @staticmethod
    def get_description():
        '''Description for visualizer legend'''
        return 'Arch A + NR + CF + SWR + STM + TKN'

    @staticmethod
    def get_dataset():
        '''Used for load the dataset and preprocess it'''
        data = Dataset()
        data.remove_noise()
        data.fold_case()
        data.remove_stopword()
        data.stem()
        data.tokenize()
        return data

    @staticmethod
    def preprocess(text):
        '''Used for preprocess the text in production environment'''
        text = Preprocessing.remove_noise(text)
        text = Preprocessing.fold_case(text)
        text = Preprocessing.remove_stopword(text)
        text = Preprocessing.stem(text)
        text = Preprocessing.tokenize(text)
        return text

    @staticmethod
    def get_network(input_dim, total_category):
        '''Used for get the architecture of this model'''
        model = Sequential()
        model.add(Embedding(
            input_dim,
            60,
            input_length=SEQUENCE_MAX_LENGTH))
        model.add(Dropout(0.2))
        model.add(Convolution1D(filters=100,
                                padding='valid',
                                kernel_size=2,
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D())
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(total_category, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', precision, recall])
        return model

class Model4:
    '''\
    This model consist of\
    Preprocessing: Noise Removal, Case Folding, Tokenization
    Architecture: Conv1D, ReLU, MaxPooling
    '''

    @staticmethod
    def get_meta():
        '''Return name of this model'''
        return 'four'

    @staticmethod
    def get_description():
        '''Description for visualizer legend'''
        return 'Arch B + NR + CF + TKN'

    @staticmethod
    def get_dataset():
        '''Used for load the dataset and preprocess it'''
        data = Dataset()
        data.remove_noise()
        data.fold_case()
        data.tokenize()
        return data

    @staticmethod
    def preprocess(text):
        '''Used for preprocess the text in production environment'''
        text = Preprocessing.remove_noise(text)
        text = Preprocessing.fold_case(text)
        text = Preprocessing.tokenize(text)
        return text

    @staticmethod
    def get_network(input_dim, total_category):
        '''Used for get the architecture of this model'''
        model = Sequential()
        model.add(Embedding(
            input_dim,
            60,
            input_length=SEQUENCE_MAX_LENGTH))
        model.add(Dropout(0.2))
        model.add(Convolution1D(filters=100,
                                padding='valid',
                                kernel_size=3,
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D())
        model.add(Convolution1D(filters=100,
                                padding='valid',
                                kernel_size=1,
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D())
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(total_category, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', precision, recall])
        return model

class Model5:
    '''\
    This model consist of\
    Preprocessing: Noise Removal, Case Folding, Tokenization
    Architecture: Conv1D, ReLU, MaxPooling
    '''

    @staticmethod
    def get_meta():
        '''Return name of this model'''
        return 'five'

    @staticmethod
    def get_description():
        '''Description for visualizer legend'''
        return 'Arch B + NR + CF + SWR + TKN'

    @staticmethod
    def get_dataset():
        '''Used for load the dataset and preprocess it'''
        data = Dataset()
        data.remove_noise()
        data.fold_case()
        data.remove_stopword()
        data.tokenize()
        return data

    @staticmethod
    def preprocess(text):
        '''Used for preprocess the text in production environment'''
        text = Preprocessing.remove_noise(text)
        text = Preprocessing.fold_case(text)
        text = Preprocessing.stem(text)
        text = Preprocessing.tokenize(text)
        return text

    @staticmethod
    def get_network(input_dim, total_category):
        '''Used for get the architecture of this model'''
        model = Sequential()
        model.add(Embedding(
            input_dim,
            60,
            input_length=SEQUENCE_MAX_LENGTH))
        model.add(Dropout(0.2))
        model.add(Convolution1D(filters=100,
                                padding='valid',
                                kernel_size=3,
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D())
        model.add(Convolution1D(filters=100,
                                padding='valid',
                                kernel_size=1,
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D())
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(total_category, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', precision, recall])
        return model

class Model6:
    '''\
    This model consist of\
    Preprocessing: Noise Removal, Case Folding, Tokenization
    Architecture: Conv1D, ReLU, MaxPooling
    '''

    @staticmethod
    def get_meta():
        '''Return name of this model'''
        return 'six'

    @staticmethod
    def get_description():
        '''Description for visualizer legend'''
        return 'Arch B + NR + CF + SWR + STM + TKN'

    @staticmethod
    def get_dataset():
        '''Used for load the dataset and preprocess it'''
        data = Dataset()
        data.remove_noise()
        data.fold_case()
        data.remove_stopword()
        data.stem()
        data.tokenize()
        return data

    @staticmethod
    def preprocess(text):
        '''Used for preprocess the text in production environment'''
        text = Preprocessing.remove_noise(text)
        text = Preprocessing.fold_case(text)
        text = Preprocessing.remove_stopword(text)
        text = Preprocessing.stem(text)
        text = Preprocessing.tokenize(text)
        return text

    @staticmethod
    def get_network(input_dim, total_category):
        '''Used for get the architecture of this model'''
        model = Sequential()
        model.add(Embedding(
            input_dim,
            60,
            input_length=SEQUENCE_MAX_LENGTH))
        model.add(Dropout(0.2))
        model.add(Convolution1D(filters=100,
                                padding='valid',
                                kernel_size=3,
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D())
        model.add(Convolution1D(filters=100,
                                padding='valid',
                                kernel_size=1,
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D())
        model.add(Dropout(0.8))
        model.add(Flatten())
        model.add(Dense(total_category, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', precision, recall])
        return model
