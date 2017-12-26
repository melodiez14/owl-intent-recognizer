import numpy as np
from data import Dataset
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Convolution1D, Embedding
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import shuffle

SEQUENCE_MAX_LENGTH = 10
NUM_EPOCHS = 150

def create_network():
    model = Sequential()
    model.add(Embedding(
                600,
                60,
                input_length=SEQUENCE_MAX_LENGTH)) # initializer, regularizer, constraint
    model.add(Dropout(0.2))
    model.add(Convolution1D(filters=60, # tunning
                            padding="valid",
                            kernel_size=2, # tunning
                            activation="relu",
                            strides=1))
    model.add(MaxPooling1D())
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(4, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

def train():

    # load dataset and preprocess
    dt = Dataset()
    dt.remove_noise()
    dt.fold_case()
    # dt.remove_stopword()
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

    print('Welcome to owl assistant training')
    print('Total data: {}'.format(len(x_input)))
    print('Total corpus: {}'.format(len(dt.get_corpus())))

    scikit_model = KerasClassifier(build_fn=create_network, 
                    epochs=10,
                    batch_size=10, 
                    verbose=2)

    cv = KFold(n_splits=4)
    cross_val_score(scikit_model, x_input, y_input, cv=cv, groups=groups)
    
if __name__ == '__main__':
    train()