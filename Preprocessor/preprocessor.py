import pyhanlp
from collections import Counter
from tqdm import tqdm
import gensim
import numpy as np
import os
import pickle as pk
import shutil
import json


def segment(input, to_string=False):
    input = list(pyhanlp.HanLP.segment(input))
    if to_string:
        return ' '.join([i.toString().split('/')[0] for i in input])
    else:
        return [i.toString().split('/')[0] for i in input]



class Preprocessor(object):
    """
    class for clean raw,get vocabulary,get trainset
    """
    def __init__(self):
        self.raw_list = ['raw/sample', 'raw/sample']
        #self.raw_list = ['raw/wiki_00', 'raw/wiki_01']
        self.proessed_root = 'processed/'
        self.counter = Counter()
        self.vocabulary = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.fast_read_w2v = 'raw/word2vec_file/sgns.wiki.fastreading.m'
        self.save_path = 'Preprocessor/preprocessor.pkl'

    def get_corpus(self):
        """
        part of pipeline_process_raw
        """
        with open('raw/corpus.txt', 'w') as writer:
            for file in tqdm(self.raw_list):
                with open(file) as reader:
                    for line in tqdm(reader, desc='get_corpus'):
                        if line != '\n':
                            lines = line.split('。')
                            for i in lines:
                                if i != '\n':
                                    words = segment(i, True).strip()
                                    if len(words.split(' ')) > 5:
                                        writer.write('<SOS>' + ' ' + words + ' ' + '。 <EOS>' + '\n')


    def get_corpus_counter(self):
        """
        part of pipeline_process_raw
        """
        with open('raw/corpus.txt') as reader:
            for i in tqdm(reader, desc='get_corpus_counter'):
                self.counter.update(Counter(str.strip(i).split(' ')))

    def pipeline_process_raw(self):
        """
        process raw file, droping blanklines, segmenting
        """
        self.get_corpus()
        self.get_corpus_counter()

    def pipeline_gen_w2v(self, drop_old_fast_read=False):
        if drop_old_fast_read:
            os.remove(self.fast_read_w2v)

        if not os.path.exists(self.fast_read_w2v):

            print('load raw w2v file and gen fast_read_w2v')
            model = gensim.models.KeyedVectors.load_word2vec_format('raw/word2vec_file/sgns.wiki.bigram')
            dic_model = {}
            for i in model.index2word:
                dic_model[i] = model[i]
            pk.dump(dic_model, open(self.fast_read_w2v, 'wb'))
        else:
            print('using fast_read_w2v')
            model = pk.load(open(self.fast_read_w2v, 'rb'))

        count_in = 0
        count_notin = 0
        bounds = np.random.uniform(-1, 1, 300).tolist()
        matrix = [np.zeros(300).tolist()] + [bounds] * 3
        for index, word in tqdm(enumerate(self.counter), desc='rebuild embedding matrix'):
            try:
                matrix.append(model[word].tolist())
                self.vocabulary[word] = index+4
                count_in += 1
            except:
                count_notin += 1
        print(f'{count_in} in found')
        print(f'{count_notin} not found')
        self.matrix = np.array(matrix)

    def pipeline_gen_trainset(self, drop_old_dataset=False):
        if os.path.exists(self.proessed_root):
            shutil.rmtree(self.proessed_root)
        os.mkdir(self.proessed_root)

        with open('raw/corpus.txt') as reader:
            for index, line in tqdm(enumerate(reader), desc='gen_trainset'):
                name = self.proessed_root + str(index) + '.json'
                line_data = {}
                tokens = line.strip().split(' ')
                ids = [self.token2id(i) for i in tokens]
                line_data['tokens'] = tokens
                line_data['ids'] = ids
                json.dump(line_data, open(name, 'w', encoding='utf8'))

    def token2id(self, token):
        try:
            return self.vocabulary[token]
        except:
            return 1

    def save(self, path=None):
        if path is None:
            pk.dump(self, open(self.save_path, 'wb'))
        else:
            pk.dump(self, open(path, 'wb'))





def test():
    pr = Preprocessor()
    pr.pipeline_process_raw()
    pr.pipeline_gen_w2v()
    pr.pipeline_gen_trainset()
    pr.save()

if __name__ == '__main__':
    test()