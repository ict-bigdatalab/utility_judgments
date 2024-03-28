import argparse
import collections
import json
import copy
import os
import re
import logging
import string
import regex
import unicodedata
from tqdm import tqdm
import numpy as np
# from rouge import Rouge
# from nltk import PorterStemmer

# stemmer = PorterStemmer()
logger = logging.getLogger()


def rouge_calculation(hypotheses, references):
	rouge = Rouge()
	scores = rouge.get_scores(hypotheses, references)
	return scores

def prepare(hypotheses, references):
	hypoth = [" ".join([stemmer.stem(i) for i in line.split()]) for line in hypotheses]
	ref = [" ".join([stemmer.stem(i) for i in line.split()]) for line in references]
	return hypoth, ref

class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
            True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                for s in range(len(words))
                for e in range(s, min(s + n, len(words)))
                if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                        (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


def has_in_ground(answers, query, text, passages, sentences=None):
    tokenizer = SimpleTokenizer()
    if has_answer(answers, text):
        for passage in passages:
            for p in passage:
                for content in p:
                    clean_hypoth, clean_ref = prepare([text], [content])
                    try:
                        score = rouge_calculation(clean_hypoth, clean_ref)
                        rouge_l = score[0]['rouge-1']['f']
                    except:
                        rouge_l = 0
                    if rouge_l > 0.30:
                        return 1
    text = unicodedata.normalize('NFD', text)
    text = tokenizer.tokenize(text).words(uncased=True)
    for passage in passages:
        for p in passage:
            for content in p:
                single_answer = unicodedata.normalize('NFD', content)
                single_answer = tokenizer.tokenize(single_answer)
                single_answer = single_answer.words(uncased=True)
                if len(text) >= len(single_answer):
                    for i in range(0, len(text) - len(single_answer) + 1):
                        if single_answer == text[i: i + len(single_answer)]:
                            return 1
                else:
                    for i in range(0, len(single_answer) - len(text) + 1):
                        if text == single_answer[i: i + len(text)]:
                            return 1
    return 0



def has_in_ground_nq(answers, query, text, passage, sentences=None):
    tokenizer = SimpleTokenizer()
    if has_answer(answers, text):
        
        clean_hypoth, clean_ref = prepare([text], [passage])

        try:
            score = rouge_calculation(clean_hypoth, clean_ref)
            rouge_l = score[0]['rouge-1']['f']
        except:
            rouge_l = 0
        if rouge_l > 0.30:
            return 1

    text = unicodedata.normalize('NFD', text)
    text = tokenizer.tokenize(text).words(uncased=True)
    
    single_answer = unicodedata.normalize('NFD', passage)
    single_answer = tokenizer.tokenize(single_answer)
    single_answer = single_answer.words(uncased=True)
    if len(text) >= len(single_answer):
        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                return 1
    else:
        for i in range(0, len(single_answer) - len(text) + 1):
            if text == single_answer[i: i + len(text)]:
                return 1
    return 0



def has_answer(answers, text, match_type="string"):
    tokenizer = SimpleTokenizer()
    text = unicodedata.normalize('NFD', text)
    if match_type == 'string':
        text = tokenizer.tokenize(text).words(uncased=True)
        for single_answer in answers:
            single_answer = unicodedata.normalize('NFD', single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i+ len(single_answer)]:
                    return 1
    return 0


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM_compute(answer_list, prediction):
    return max([int(_normalize_answer(prediction) == _normalize_answer(ground_truth)) for ground_truth in answer_list])

def EM_compute_tq(answer_list, prediction):
    return max([int(_normalize_answer(prediction) == _normalize_answer(ground_truth)) for ground_truth in answer_list])

def F1_compute(answers, pred):
    def get_tokens(s):
        if not s: return []
        return _normalize_answer(s).split()

    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    return max([compute_f1(x, pred) for x in answers])


def deal_judge(pred):
    if pred is None:
        return True
    if has_answer(["unknown", "no specific answer", "not provide", "cannot answer", "no information provided", "no answer", "not contain", "no definitive answer"], pred):
        return True
    return False


def deal_answer(pred, answers):
    if pred is None:
        return 0, 0
    pred = pred[0]
    return EM_compute(answers, pred), F1_compute(answers, pred)
        

# def deal_post(pred):
#     giveup, istrue = True, None
#     if pred is None:
#         return giveup, istrue
#     if has_answer(["unclear", "not clear", "unknown", "partially correct", "partially incorrect", "not correct", "cannot determine", "cannot answer", "not incorrect", "incomplete"], pred):
#         giveup = True
#     elif has_answer(["correct", "true"], pred):
#         giveup, istrue = False, True
#     elif has_answer(["incorrect", "false"], pred):
#         giveup, istrue = False, False
#     else:
#         giveup = True
#     return giveup, istrue


def str2paras(s):
        if s is None:
            return None
        paras = []
        for text in s.split('\n'):
            if text.strip() != '':
                paras.append(": " + text)
        return paras


if __name__ == "__main__":
    root = "D:/LLM-Knowledge-Boundary-main/data/introduction"
    # file_list = [root+"/chatgpt-nq-gold", root+"/chatgpt-tq_final-gold", root+"/chatgpt-hq_final-gold"]
    # file_list = [root + "/chatgpt-nq-dense", root + "/chatgpt-tq_final-dense", root + "/chatgpt-hq_final-dense"]
    file_list = [root + "/chatgpt-nq-none", root + "/chatgpt-tq_final-none", root + "/chatgpt-hq_final-none"]
    for file in file_list:
        em_scores = []
        f1_scores = []
        outdir = open(file+"_results.json", "w", encoding="utf-8")
        infile = open(file+".json", 'r', encoding='utf-8')
        for line in tqdm(infile.readlines()):
            d = json.loads(line)
            d['EM'], d['F1'] = deal_answer(d['model_answer'], d['ground_answer'])
            outdir.write(json.dumps(d) + '\n')
            em_scores.append(d['EM'])
            f1_scores.append(d['F1'])

        infile.close()
        outdir.close()
        em_scores_mean = np.mean(em_scores)
        f1_scores_mean = np.mean(f1_scores)
        print("{}'s em score: {}".format(file, em_scores_mean))
        print("{}'s f1 score: {}".format(file, f1_scores_mean))


def load_source(file):
    data = []
    f = open(file, 'r', encoding='utf-8')
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    return data
