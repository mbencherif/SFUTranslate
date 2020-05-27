import glob
import os
import io
import codecs
import xml.etree.ElementTree as ET
from torchtext import data, datasets
from configuration import src_lan, tgt_lan, cfg, device


class Multi30k(datasets.TranslationDataset):
    """The small-dataset WMT 2017 multimodal task"""

    urls = ['http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_training.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_validation.tar.gz',
            'http://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/mmt_task1_test2016.tar.gz']
    name = 'multi30k'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='val', test='test2016', **kwargs):
        """Create dataset objects for splits of the Multi30k dataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(Multi30k, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)


class IWSLT(datasets.TranslationDataset):
    """Do-over of the original Torchtext IWSLT Library.
    This one just does not apply filter_pred designed for length limiting to the test and dev datasets.
    The IWSLT 2016 TED talk translation task"""

    # base_url = 'https://wit3.fbk.eu/archive/2016-01//texts/{}/{}/{}.tgz'
    base_url = 'https://wit3.fbk.eu/archive/2017-01-trnted/texts/{}/{}/{}.tgz'
    name = 'iwslt'
    base_dirname = '{}-{}'

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='IWSLT17.TED.dev2010',
               test_list=('IWSLT17.TED.tst2010', 'IWSLT17.TED.tst2011', 'IWSLT17.TED.tst2012', 'IWSLT17.TED.tst2013',
                          'IWSLT17.TED.tst2014', 'IWSLT17.TED.tst2015'), **kwargs):
        debug_mode = False
        if "debug_mode" in kwargs:
            debug_mode = kwargs["debug_mode"]
            del kwargs["debug_mode"]
        cls.dirname = cls.base_dirname.format(exts[0][1:], exts[1][1:])
        cls.urls = [cls.base_url.format(exts[0][1:], exts[1][1:], cls.dirname)]
        check = os.path.join(root, cls.name, cls.dirname)
        path = cls.download(root, check=check)

        if train is not None:
            train = '.'.join([train, cls.dirname])
        validation = '.'.join([validation, cls.dirname])
        tests = []
        if test_list is not None:
            for t in test_list:
                tests.append('.'.join([t, cls.dirname]))
        if train is not None:
            if not os.path.exists(os.path.join(path, train) + exts[0]):
                cls.clean(path)
            print("    [torchtext] Loading train examples ...")
            train_data = cls(os.path.join(path, train), exts, fields, **kwargs)
        else:
            train_data = None
        # Here is the line that have been added.
        if not debug_mode:
            kwargs['filter_pred'] = None
        print("    [torchtext] Loading validation examples ...")
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        val_data.name = validation
        print("    [torchtext] Loading test examples ...")
        test_data_list = [None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs) for test in tests]
        if len(tests):
            for d, n in zip(test_data_list, test_list):
                d.name = n
        return tuple(d for d in (train_data, val_data, *test_data_list)
                     if d is not None)

    @staticmethod
    def clean(path):
        for f_xml in glob.iglob(os.path.join(path, '*.xml')):
            print(f_xml)
            f_txt = os.path.splitext(f_xml)[0]
            with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt:
                root = ET.parse(f_xml).getroot()[0]
                for doc in root.findall('doc'):
                    for e in doc.findall('seg'):
                        fd_txt.write(e.text.strip() + '\n')

        xml_tags = ['<url', '<keywords', '<talkid', '<description',
                    '<reviewer', '<translator', '<title', '<speaker']
        for f_orig in glob.iglob(os.path.join(path, 'train.tags*')):
            print(f_orig)
            f_txt = f_orig.replace('.tags', '')
            with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt, \
                    io.open(f_orig, mode='r', encoding='utf-8') as fd_orig:
                for l in fd_orig:
                    if not any(tag in l for tag in xml_tags):
                        fd_txt.write(l.strip() + '\n')


class WMT19DeEn(datasets.TranslationDataset):
    """The WMT 2019 English-German dataset, processed using the script in
    https://drive.google.com/open?id=1rYNpc2VNXXGINPLd2CZvH3lf_w19O8-r"""

    urls = [('https://drive.google.com/uc?export=download&'
             'id=1miCyP1Vdoi6QsGoKSWhNgxbPVmSSs3Rt', 'wmt19_en_de_raw.zip')]
    name = 'wmt19_en_de'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data', train='train', validation='valid',
               test_list=('newstest2014', 'newstest2015', 'newstest2016',
                          'newstest2017', 'newstest2018', 'newstest2019'), **kwargs):
        """Create dataset objects for splits of the WMT 2014 dataset.
        Arguments:
            exts: A tuple containing the extensions for each language. Must be
                either ('.en', '.de') or the reverse.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data.
            validation: The prefix of the validation data.
            test_list: The prefix of the test data.
            Remaining keyword arguments: Passed to the splits method of Dataset.
        """
        debug_mode = False
        if "debug_mode" in kwargs:
            debug_mode = kwargs["debug_mode"]
            del kwargs["debug_mode"]
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        if path is None:
            path = cls.download(root)
        if train is not None:
            print("    [torchtext] Loading train examples ...")
        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        kwargs['filter_pred'] = None
        print("    [torchtext] Loading validation examples ...")
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        val_data.name = validation
        print("    [torchtext] Loading test examples ...")
        test_data_list = [None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs) for test in test_list]
        if len(test_list):
            for d, n in zip(test_data_list, test_list):
                d.name = n
        return tuple(d for d in (train_data, val_data, *test_data_list)
                     if d is not None)


class WMT19DeFr(datasets.TranslationDataset):
    """The WMT 2019 English-French dataset, processed using the script in
    https://drive.google.com/open?id=1-HJr69Z-Svl55xo5c2fco7QOCXeETh0H"""

    urls = [('https://drive.google.com/uc?export=download&'
             'id=1-HJr69Z-Svl55xo5c2fco7QOCXeETh0H', 'wmt19_de_fr.zip')]
    name = 'wmt19_de_fr'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data', train='train', validation='valid',
               test_list=('newstest2008', 'newstest2009', 'newstest2010', 'newstest2011',
                          'newstest2012', 'newstest2013', 'newstest2019', 'euelections_dev2019'), **kwargs):
        """Create dataset objects for splits of the WMT 2019 dataset.
        Arguments:
            exts: A tuple containing the extensions for each language. Must be
                either ('.en', '.fr') or the reverse.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data.
            validation: The prefix of the validation data.
            test_list: The prefix of the test data.
            Remaining keyword arguments: Passed to the splits method of Dataset.
        """
        debug_mode = False
        if "debug_mode" in kwargs:
            debug_mode = kwargs["debug_mode"]
            del kwargs["debug_mode"]
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        if path is None:
            path = cls.download(root)
        if train is not None:
            print("    [torchtext] Loading train examples ...")
        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        kwargs['filter_pred'] = None
        print("    [torchtext] Loading validation examples ...")
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        val_data.name = validation
        print("    [torchtext] Loading test examples ...")
        test_data_list = [None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs) for test in test_list]
        if len(test_list):
            for d, n in zip(test_data_list, test_list):
                d.name = n
        return tuple(d for d in (train_data, val_data, *test_data_list)
                     if d is not None)
