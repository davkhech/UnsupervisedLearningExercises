from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import numpy as np
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
import string


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    """
    Convert a document into a list of words (also remove stop words and punctuation)
    """
    # TODO: Remove stop words
    # TODO: remove punctuation
    # TODO: Do somthing else smart
    return list([])


def build_corpus_features(cleaned_docs, min_count=2, max_frequency=0.94):
    """
    Build a dictionary from words in dictonary to their indices.
    """
    # TODO: Implement
    return {'aaa': 1, 'abc': 2}


def featurize(cleaned_doc, corpus_dict):
    features = np.zeros(len(corpus_dict), dtype=np.int)
    # TODO: Implement
    return features


def display_topics(model, index_to_word, word_to_display):
    # TODO: Write code that print the topics.
    print('TODO: Replace me')


if __name__ == '__main__':
    trainset = fetch_20newsgroups(shuffle=True, random_state=1,
                                  remove=('headers', 'footers', 'quotes'))
    train_data = trainset.data
    cleaned_docs = [clean(doc) for doc in train_data]
    corpus_dict = build_corpus_features(cleaned_docs)
    # Note: Not a very optimal code
    featurized_docs = np.array([featurize(doc, corpus_dict) for doc in cleaned_docs])  # noqa
    lda = LatentDirichletAllocation(n_topics=20, learning_method='online')
    lda.fit(featurized_docs)
    index_to_word = next(zip(*sorted(corpus_dict.items(), key=lambda x: x[1])))
    display_topics(lda, index_to_word, 21)
