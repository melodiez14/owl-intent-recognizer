'''Containing Evaluator class for evaluating model'''

import os
import numpy as np
from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

SEQUENCE_MAX_LENGTH = 10
NUM_EPOCHS = 30
EVAL_DIR = 'evaluation'
MODEL_DIR = 'model'
MODEL_FILENAME = 'trainned.h5'
MODEL_DICTIONARY_X = 'dictionary.json'
MODEL_DICTIONARY_Y = 'intent.json'


class Evaluator:
    '''This class used for evaluate all models specified'''

    def __init__(self, *args):
        self.__models__ = [arg for arg in args]

    def evaluate(self):
        '''Evaluates each model registered to the constructor'''
        from keras.preprocessing.text import Tokenizer

        if not os.path.exists(EVAL_DIR):
            os.makedirs(EVAL_DIR)

        for model in self.__models__:
            data = model.get_dataset()

            # initialize corpus
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(data.get_corpus())

            # prepare input
            x_input, y_input = [], []
            for i, value in enumerate(data.get()):
                x_input += [[tokenizer.word_index[y]
                             for y in x] for x in value]
                y_input += [i for x in value]

            # pad sequence input
            groups = y_input
            x_input = sequence.pad_sequences(
                x_input, maxlen=SEQUENCE_MAX_LENGTH, padding="post", truncating="post")
            y_input = to_categorical(y_input)

            # shuffle data
            x_input, y_input, groups = shuffle(
                x_input, y_input, groups, random_state=7)

            total_corpus = len(data.get_corpus())
            total_category = data.get_category_length()

            # prepare counted metrics
            test_accuracy, test_precision, test_recall = [], [], []
            kfold = KFold(n_splits=10, shuffle=True, random_state=7)
            for train_index, test_index in kfold.split(x_input, y_input, groups=groups):
                # splitting training and test set
                x_train = x_input[train_index]
                x_test = x_input[test_index]
                y_train = y_input[train_index]
                y_test = y_input[test_index]
                # training
                network = model.get_network(total_corpus + 1, total_category)
                fit = network.fit(x_train, y_train,
                                  epochs=NUM_EPOCHS,
                                  verbose=1,
                                  validation_data=(x_test, y_test),
                                  batch_size=100)
                # save the metrics
                test_accuracy.append(fit.history['val_acc'])
                test_precision.append(fit.history['val_precision'])
                test_recall.append(fit.history['val_recall'])

            np.save(
                '{0}/{1}_test_accuracy'.format(EVAL_DIR, model.__name__), test_accuracy)
            np.save('{0}/{1}_test_precision'.format(EVAL_DIR, model.__name__),
                    test_precision)
            np.save('{0}/{1}_test_recall'.format(EVAL_DIR,
                                                 model.__name__), test_recall)


class Trainner:
    '''This class used for evaluate all models specified'''

    def __init__(self, model):
        self.__model__ = model

    def train(self):
        '''Evaluates each model registered to the constructor'''
        from keras.preprocessing.text import Tokenizer

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        data = self.__model__.get_dataset()

        # initialize corpus
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data.get_corpus())

        # prepare input
        x_input, y_input = [], []
        for i, value in enumerate(data.get()):
            x_input += [[tokenizer.word_index[y]
                         for y in x] for x in value]
            y_input += [i for x in value]

        # pad sequence input
        groups = y_input
        x_input = sequence.pad_sequences(
            x_input, maxlen=SEQUENCE_MAX_LENGTH, padding="post", truncating="post")
        y_input = to_categorical(y_input)

        # shuffle data
        x_input, y_input, groups = shuffle(
            x_input, y_input, groups, random_state=7)

        total_corpus = len(data.get_corpus())
        total_category = data.get_category_length()

        # training
        network = self.__model__.get_network(total_corpus + 1, total_category)
        network.fit(x_input, y_input,
                    epochs=NUM_EPOCHS,
                    verbose=1,
                    batch_size=2)

        # save the dictionary
        import json
        dictionary = tokenizer.word_index
        with open(os.path.join(MODEL_DIR, MODEL_DICTIONARY_X), 'w') as dictionary_file:
            json.dump(dictionary, dictionary_file)
        
        dictionary = data.get_intent()
        with open(os.path.join(MODEL_DIR, MODEL_DICTIONARY_Y), 'w') as dictionary_file:
            json.dump(dictionary, dictionary_file)

        # save the model
        network.save('{0}/{1}'.format(MODEL_DIR, MODEL_FILENAME))
        print('''Model Successfully trainned. Now you can run the server''')


class Webserver:
    '''Webserver used for running the intent recognition service'''

    def __init__(self):
        # load model
        path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        if not os.path.exists(path):
            print(
                'Please run "python app.py train [modelnumber]" before use run application')

        from keras.models import load_model
        from metric import precision, recall
        self.__model__ = load_model(
            path, {'precision': precision, 'recall': recall})

        # load dictionary
        path = os.path.join(MODEL_DIR, MODEL_DICTIONARY_X)
        if not os.path.exists(path):
            print(
                'Please run "python app.py train [modelnumber]" before use run application')

        import json
        with open(os.path.join(MODEL_DIR, MODEL_DICTIONARY_X), 'r') as dictionary_file:
            self.__dictionary__ = json.load(dictionary_file)

        path = os.path.join(MODEL_DIR, MODEL_DICTIONARY_Y)
        if not os.path.exists(path):
            print(
                'Please run "python app.py train [modelnumber]" before use run application')

        with open(os.path.join(MODEL_DIR, MODEL_DICTIONARY_Y), 'r') as dictionary_file:
            self.__intent__ = json.load(dictionary_file)


    def run(self):
        '''This function used for running the web server and predicting the result'''

        import json
        from util import Preprocessing
        from flask import Flask, request

        app = Flask(__name__)

        @app.route('/api/v1/predict', methods=['POST'])
        def predict():
            '''route for predict the intent'''
            text = request.form['text']
            text = Preprocessing.remove_noise(text)
            text = Preprocessing.fold_case(text)
            text = Preprocessing.tokenize(text)
            # convert input to an index of dictionary
            x_input = []
            for word in text:
                index = self.__dictionary__.get(word, None)
                if index is not None:
                    x_input += [index]
            # pad sequence
            x_input = sequence.pad_sequences(
                [x_input], maxlen=SEQUENCE_MAX_LENGTH, padding="post", truncating="post")
            
            prediction = self.__model__.predict(x_input)
            result = np.argmax(prediction, axis=1)
            response = json.dumps({
                'intent': self.__intent__[result[0]],
                'confident': np.float64(prediction[0][result[0]])
            })
            return response
        
        app.run()
