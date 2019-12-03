import string
import re


def cleaning_sentences(sentences, punctuations):
    re_punc = re.compile('[%s]' % re.escape(punctuations))
    return [[re_punc.sub("", w).lower() for w in s.split() if re_punc.sub("", w).lower().isalpha()]
            for s in sentences]


def get_indexed_token(tokenized_text, word2idx):
    tokenized_indexed_text = []
    for idx in range(len(tokenized_text)):
        text = tokenized_text[idx]
        text_word_indexes = []
        text_word_indexes.append(word2idx['[CLS]'])
        for word in text:
            if word in word2idx:
                word_idx = word2idx[word]
            else:
                word_idx = word2idx['UNK']
            if word_idx > 0:
                text_word_indexes.append(word_idx)
        text_word_indexes.append(word2idx['[SEP]'])
        tokenized_indexed_text.append(text_word_indexes)

    return tokenized_indexed_text
