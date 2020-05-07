from tokenizers import BertWordPieceTokenizer
from requests import get
import spacy
import os


class PreTrainedTokenizer:
    vocab_files = {
        "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
        "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
        "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
        "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
        "bert-base-german-cased": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
        "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
        "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
        "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
        "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
        "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
        "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt",
        "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-vocab.txt",
        "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/vocab.txt",
        "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/vocab.txt",
        "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/vocab.txt",
    }

    def __init__(self, pre_trained_model_name, root='.data', clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True):
        """
        Example instantiation: PreTrainedTokenizer("bert-base-uncased", root="../.data")
        """
        if not os.path.exists(root):
            os.mkdir(root)
        assert pre_trained_model_name in self.vocab_files, \
            "The requested pre_trained tokenizer model {} does not exist!".format(pre_trained_model_name)
        url = self.vocab_files[pre_trained_model_name]
        f_name = root + "/" + os.path.basename(url)
        if not os.path.exists(f_name):
            with open(f_name, "wb") as file_:
                response = get(url)
                file_.write(response.content)
        self.tokenizer = BertWordPieceTokenizer(f_name, clean_text=clean_text, lowercase=lowercase,
                                                handle_chinese_chars=handle_chinese_chars, strip_accents=strip_accents)

    def tokenize(self, text):
        """
        You can recover the output of this function using " ".join(encoded_list).replace(" ##", "")
        :param text: one line of text in type of str
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        # print(f"IDS: {encoding.ids}")
        # print(f"TOKENS: {encoding.tokens}")
        # print(f"OFFSETS: {encoding.offsets}")
        return encoding.tokens

    def decode(self, encoded_ids_list):
        """
        :param encoded_ids_list: list of int ids
        """
        decoded = self.tokenizer.decode(encoded_ids_list)
        # print(f"DECODED: {decoded}")
        return decoded

    @staticmethod
    def get_default_model_name(lang, lowercase):
        if lang == "en" and lowercase:
            return "bert-base-uncased"
        elif lang == "en" and not lowercase:
            return "bert-base-cased"
        elif lang == "zh":
            return "bert-base-chinese"
        elif lang == "de" and lowercase:
            return "bert-base-german-dbmdz-uncased"
        elif lang == "de" and not lowercase:
            return "bert-base-german-dbmdz-cased"
        elif lang == "fi" and lowercase:
            return "bert-base-finnish-uncased-v1"
        elif lang == "fi" and not lowercase:
            return "bert-base-finnish-cased-v1"
        else:
            raise ValueError("No pre-trained tokenizer found for language {} in {} mode".format(
                lang, "lowercased" if lowercase else "cased"))


class SpacyTokenizer:
    def __init__(self, tokeniztion_lang):
        self.tokenizer = spacy.load(tokeniztion_lang)

    def tokenize(self, text):
        return [tok.text for tok in self.tokenizer.tokenizer(text)]


class SplitTokenizer:
    @staticmethod
    def tokenize(text):
        return text.split()


def get_tokenizer_from_configs(tokenizer_name, lang, lowercase_data, debug_mode=False):
    print("Loading tokenizer of type {} for {} language".format(tokenizer_name, lang))
    if tokenizer_name == "spacy":
        return SpacyTokenizer(lang)
    elif tokenizer_name == "split" or bool(debug_mode):
        return SplitTokenizer()
    elif tokenizer_name == "pre_trained":
        return PreTrainedTokenizer(PreTrainedTokenizer.get_default_model_name(lang, lowercase_data))
    else:
        raise ValueError("The requested tokenizer {} does not exist or is not implemented!".format(tokenizer_name))
