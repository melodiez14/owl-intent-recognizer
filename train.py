import time
import numpy as np
import matplotlib.pyplot as plt
from data import Dataset
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Convolution1D, Embedding
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from metric import precision, recall

SEQUENCE_MAX_LENGTH = 10
NUM_EPOCHS = 150

def create_network(total_category, input_dim = 1000):
    model = Sequential()
    model.add(Embedding(
                input_dim,
                60,
                input_length=SEQUENCE_MAX_LENGTH)) # initializer, regularizer, constraint
    model.add(Dropout(0.2))
    model.add(Convolution1D(filters=10, # tunning
                            padding="valid",
                            kernel_size=2, # tunning
                            activation="relu",
                            strides=1))
    model.add(MaxPooling1D())
    model.add(Convolution1D(filters=100, # tunning
                            padding="valid",
                            kernel_size=1, # tunning
                            activation="relu",
                            strides=1))
    model.add(MaxPooling1D())
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(total_category, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy', precision, recall])
    return model

def train():

    # load dataset and preprocess
    dt = Dataset()
    dt.remove_noise()
    dt.fold_case()
    dt.remove_stopword()
    # dt.stem()
    dt.tokenize()

    # initialize corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dt.get_corpus())

    # prepare input
    x_input = []
    y_input = []
    for i, value in enumerate(dt.get()):
        x_input += [[tokenizer.word_index[y] for y in x] for x in value]
        y_input += [i for x in value]

    # pad sequence input
    groups = y_input
    x_input = sequence.pad_sequences(x_input, maxlen=SEQUENCE_MAX_LENGTH, padding="post", truncating="post")
    y_input = to_categorical(y_input)

    # shuffle data
    x_input, y_input, groups = shuffle(x_input, y_input, groups, random_state=7)

    total_corpus = len(dt.get_corpus())
    total_category = dt.get_category_length()

    print('Welcome to owl assistant training')
    print('Total data: {}'.format(len(x_input)))
    print('Total corpus: {}'.format(total_corpus))
    print('Total category: {}'.format(total_category))

    model = create_network(total_category, total_corpus+1)
    kfold = KFold(n_splits=4, shuffle=True, random_state=7)
    train_accuracy = []
    train_precision = []
    train_recall = []
    test_accuracy = []
    test_precision = []
    test_recall = []
    t1 = time.time()
    for train_index, test_index in kfold.split(x_input, y_input, groups=groups):
        x_train = x_input[train_index]
        x_test = x_input[test_index]
        y_train = y_input[train_index]
        y_test = y_input[test_index]
        fit = model.fit(x_train, y_train,
                epochs=100,
                verbose=2,
                validation_data=(x_test, y_test),
                batch_size=10)
        train_accuracy += fit.history['acc']
        train_precision += fit.history['precision']
        train_recall += fit.history['recall']
        test_accuracy += fit.history['val_acc']
        test_precision += fit.history['val_precision']
        test_recall += fit.history['val_recall']
    
    print(time.time() - t1)
    model.save('save/model_5.h5')
    np.save('save/model_5_train_accuracy', train_accuracy)
    np.save('save/model_5_train_precision', train_precision)
    np.save('save/model_5_train_recall', train_recall)
    np.save('save/model_5_test_accuracy', test_accuracy)
    np.save('save/model_5_test_precision', test_precision)
    np.save('save/model_5_test_recall', test_recall)

if __name__ == '__main__':
    train()