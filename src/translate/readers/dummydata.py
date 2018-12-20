"""
Provides a generated pack of sentences which mimics the task of copy + reverse in neural machine translation, e.g. for a
 sentence "a b c d e f a r" it should provide "r a f e d c b a" as the translation of the instance. To make your model
  use this reader, set "reader.dataset.type" in your config file to "dummy" and set the following values in it with your
    desired values:
####################################
    reader:
        dataset:
            type: dummy
            dummy:
                min_len: 8
                max_len: 50
                vocab_size: 96
                train_samples: 40000
                test_samples: 3000
                dev_samples: 1000
    trainer:
        experiment:
            name: 'dummy'
####################################
"""
import string
from random import randint

from translate.configs.loader import ConfigLoader
from translate.readers.constants import ReaderType
from translate.readers.datareader import AbsDatasetReader

__author__ = "Hassan S. Shavarani"


class DummyDataset(AbsDatasetReader):
    def __init__(self, configs: ConfigLoader, reader_type: ReaderType):
        """
        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param reader_type: an intance of ReaderType enum stating the type of the dataste (e.g. Train, Test, Dev)
        """
        super().__init__(configs, reader_type)
        self.min_length = configs.get("reader.dataset.dummy.min_len", must_exist=True)
        self.max_length = configs.get("reader.dataset.dummy.max_len", must_exist=True)
        self.vocab_size = configs.get("reader.dataset.dummy.vocab_size", must_exist=True)
        if reader_type == ReaderType.TRAIN:
            self.max_samples = configs.get("reader.dataset.dummy.train_samples", must_exist=True)
        elif reader_type == ReaderType.TEST:
            self.max_samples = configs.get("reader.dataset.dummy.test_samples", must_exist=True)
        elif reader_type == ReaderType.DEV:
            self.max_samples = configs.get("reader.dataset.dummy.dev_samples", must_exist=True)
        else:
            self.max_samples = 1
        # The desired vocabulary given the vocab_size set in config file gets created in here
        #  and is set inside source and target vocabulry objects
        tmp = [x for x in string.ascii_letters + string.punctuation + string.digits]
        vocab = [x + "," + y for x in tmp for y in tmp if x != y][:self.vocab_size]
        self.source_vocabulary.set_types(vocab)
        self.target_vocabulary.set_types(vocab)
        if self.max_samples > 1:
            self.pairs = [self._get_next_pair() for _ in range(self.max_samples)]
        else:
            self.pairs = [self._get_next_pair(self.max_length)]
        self.reading_index = 0

    def max_sentence_length(self):
        return self.max_length

    def __next__(self):
        """
        The function always iterates over the already generated/cached pairs of sequences (with their reverse sequence)
        """
        if self.reading_index < len(self.pairs):
            tmp = self.pairs[self.reading_index]
            self.reading_index += 1
            return tmp
        else:
            self.reading_index = 0
        raise StopIteration

    def __getitem__(self, idx):
        return self.pairs[idx]

    def __len__(self):
        return len(self.pairs)

    def allocate(self):
        return

    def deallocate(self):
        return

    def _get_next_pair(self, expected_length=0):
        """
        The function which given an :param expected_length: size of sentence, creates a random string from the
         vocabulary and returns it along with its reverse in return
        """
        if expected_length == 0:
            expected_length = randint(self.min_length - 1, self.max_length - 1)
        if expected_length >= self.max_length:
            expected_length = self.max_length - 1
        tmp = [self.target_vocabulary[x] for x in self.target_vocabulary.retrieve_dummy_words_list(expected_length)]
        rev = [x for x in tmp[::-1]]
        tmp += [self.target_vocabulary.get_end_word_index()]
        rev += [self.target_vocabulary.get_end_word_index()]
        return tmp, rev
