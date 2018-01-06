import os
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class Dataset:

    QUESTION_DATASET_PATH = [
        'dataset/assignment.txt',
        'dataset/grade.txt',
        'dataset/assistant.txt',
        'dataset/schedule.txt',
        'dataset/information.txt'
    ]
   
    def __init__(self):
        self.__texts__ = self.__init_text__()
        self.__corpus__ = []
        self.__category_length__ = len(self.QUESTION_DATASET_PATH)

    def __init_text__(self):
        plain = []
        for path in self.QUESTION_DATASET_PATH:
            if not os.path.isfile(path):
                print('load corpus failed')
                print('%s is not exist' % path)
                exit()

            with open(path, 'r') as file:
                plain += [[x for x in file.read().split('\n')]]
        return plain

    def remove_noise(self):
        texts = []
        for text in self.__texts__:
            texts += [[re.sub(r'[^A-Za-z ]', '', x) for x in text]]
        self.__texts__ = texts

    def fold_case(self):
        texts = []
        for text in self.__texts__:
            texts += [[x.lower() for x in text]]
        self.__texts__ = texts
    
    def tokenize(self):
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
        f = StemmerFactory()
        stemmer = f.create_stemmer()
        texts = []
        for text in self.__texts__:
            texts += [[stemmer.stem(x) for x in text]]
        self.__texts__ = texts

    def remove_stopword(self):
        f  = StopWordRemoverFactory()
        remover = f.create_stop_word_remover()
        texts = []
        for text in self.__texts__:
            texts += [[remover.remove(x) for x in text]]
        self.__texts__ = texts

    def get(self):
        return self.__texts__

    def get_corpus(self):
        return self.__corpus__

    def get_category_length(self):
        return self.__category_length__


if __name__ == '__main__':
    dt = Dataset()
    dt.remove_noise()
    dt.fold_case()
    # dt.remove_stopword()
    dt.stem()
    dt.tokenize()
    print('Total category\t: {0}'.format(len(dt.get())))
    print('Total corpus\t: {0}'.format(len(dt.get_corpus())))
    