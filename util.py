'''This module used for the util like dataset loader and preprocessing'''


class Preprocessing:
    '''Class contains all used preprocessing'''
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    __stemmer__ = StemmerFactory().create_stemmer()
    __remover__ = StopWordRemoverFactory().create_stop_word_remover()

    @staticmethod
    def remove_noise(text):
        '''Delete all character except alphabet and space'''
        import re
        return re.sub(r'[^A-Za-z ]', '', text)

    @staticmethod
    def fold_case(text):
        '''Make the text lowercase'''
        return text.lower()

    @staticmethod
    def tokenize(text):
        '''Split the text into word'''
        return text.split()

    @classmethod
    def stem(cls, text):
        '''Change each word to be the base or remove the preposition'''
        return cls.__stemmer__.stem(text)

    @classmethod
    def remove_stopword(cls, text):
        '''Remove conjunction and unmeaning to any class'''
        return cls.__remover__.remove(text)


class Dataset:
    '''Used for load and preprocess dataset'''

    QUESTION_DATASET_PATH = {
        'assignment': 'dataset/assignment.txt',
        'grade': 'dataset/grade.txt',
        'assistant': 'dataset/assistant.txt',
        'schedule': 'dataset/schedule.txt',
        'information': 'dataset/information.txt'
    }

    def __init__(self):
        self.__texts__ = self.__init_text__()
        self.__corpus__ = []
        self.__category_length__ = len(self.QUESTION_DATASET_PATH)

    def __init_text__(self):
        import os
        plain = []
        for _, path in self.QUESTION_DATASET_PATH.items():
            if not os.path.isfile(path):
                print('load corpus failed')
                print('%s is not exist' % path)
                exit()

            with open(path, 'r') as file:
                plain += [[x for x in file.read().split('\n')]]
        return plain

    def remove_noise(self):
        '''Delete all character except alphabet and space'''
        texts = []
        for text in self.__texts__:
            texts += [[Preprocessing.remove_noise(x) for x in text]]
        self.__texts__ = texts

    def fold_case(self):
        '''Make the text lowercase'''
        texts = []
        for text in self.__texts__:
            texts += [[Preprocessing.fold_case(x) for x in text]]
        self.__texts__ = texts

    def tokenize(self):
        '''Split the text into word and get the corpus'''
        texts = []
        corpus = []
        # tokenize
        for text in self.__texts__:
            texts += [[x.split() for x in text]]
            for x in text:
                corpus += x.split()

        # select unique corpus
        self.__corpus__ = list(set(corpus))
        self.__texts__ = texts

    def stem(self):
        '''Change each word to be the base or remove the preposition'''
        texts = []
        for text in self.__texts__:
            texts += [[Preprocessing.stem(x) for x in text]]
        self.__texts__ = texts

    def remove_stopword(self):
        '''Remove conjunction and unmeaning to any class'''
        texts = []
        for text in self.__texts__:
            texts += [[Preprocessing.remove_stopword(x) for x in text]]
        self.__texts__ = texts

    def get(self):
        '''Return the dataset'''
        return self.__texts__

    def get_corpus(self):
        '''Return the corpuses contained in dataset'''
        return self.__corpus__

    def get_category_length(self):
        '''Return total intent'''
        return self.__category_length__

    def get_intent(self):
        '''Return an intent'''
        return [x for x in self.QUESTION_DATASET_PATH]
