import math
from collections import Counter
from pathlib import Path
import random
from statistics import mean, harmonic_mean

import numpy
from bertopic import BERTopic
from pandas import DataFrame

import numpy as np
import pandas as pd
from scipy.stats import entropy, hmean
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Fix Python and numpy seeds
random.seed(42)
np.random.seed(42)

def mean_topic_entropy_score(topic_entropies: list, weights) -> float:
    print(weights)
    #print(hmean(topic_entropies, weights=weights))
    #print(hmean(topic_entropies, weights=weights)/ math.log(len(topic_entropies),2))
    #print(numpy.average(topic_entropies, weights=weights) / math.log(len(topic_entropies), 2))
    #print(mean(topic_entropies))
    return mean(topic_entropies)/ math.log(len(topic_entropies),2)

def topic_shannon_entropy(predictions, labels):
    '''Calculate Shannon information entropy of a topic. Assumes that anomaly yes/no is binary.'''
    print(len(predictions), len(labels))

    binary_labels = [int(label != "-") for label in labels]

    print(Counter(labels))

    topic_label_matrix = {}
    for idx, document_assignment in enumerate(predictions):
        if document_assignment not in topic_label_matrix:
            topic_label_matrix[document_assignment] = []
        topic_label_matrix[document_assignment].append(binary_labels[idx])

    entropies = []
    weights = []
    for topic, distribution in topic_label_matrix.items():
        freq_counts = Counter(distribution)

        # Contains at least one anomaly - we are interested in the minority class
        amount_of_documents = len(distribution)
        topic_binary_shannon_entropy = entropy([freq_counts[0], freq_counts[1]], base=2)

        print(topic)
        print(freq_counts, amount_of_documents)
        print(topic_binary_shannon_entropy)

        # We weigh clusters by the proportion of anomalies as we are interested in those.
        # Intuition: we don't learn anything from non-anomalous topics
        # The score should increase if there are very pure high-anomaly topics
        # Note that this comparison only works within a dataset (extremely sensitive to class balances).
        # New problem: score decreases if
        weight = freq_counts[1]/len(distribution)
        if weight != 0:
            weights.append(weight)
            #entropies.append(topic_binary_shannon_entropy)
            entropies.append(freq_counts[1]/len(distribution))

        #weights.append(1/len(distribution))

    print(entropies)
    print(len(entropies))
    print(mean_topic_entropy_score(entropies, weights))

def main():

    thunderbird_df_train = pd.read_csv("thunderbird_train.tsv",sep="\t")
    thunderbird_df_test =  pd.read_csv("thunderbird_test.tsv", sep="\t")
    bgl_df_train =  pd.read_csv("bgl_train.tsv", sep="\t")
    bgl_df_test =  pd.read_csv("bgl_test.tsv", sep="\t")

    bgl_df_train = thunderbird_df_train

    print(thunderbird_df_train)

    # On all data combined - poor classifier but good explainer?

    # Supervised by all classes (no need to change metric; binary is easier than multiclass so its actually quite lenient)
    # Supervised anomaly yes/no
    # Unsupervised
    # Unsupervised - LDA

    # Within dataset x2
    # Across dataset x2

    thunderbird_label_encoder = LabelEncoder().fit(bgl_df_train["Label"])

    vectorizer_model = CountVectorizer(max_df=1.0, min_df=1, ngram_range=(1, 1), strip_accents='unicode',
                                       stop_words='english', lowercase=False)

    topic_model = BERTopic(verbose=True, vectorizer_model=vectorizer_model, calculate_probabilities=True,
                           top_n_words=-1)
    #predictions, distributions = topic_model.fit_transform(bgl_df_train["ComponentEventTemplate"], y=thunderbird_label_encoder.transform(bgl_df_train["Label"]))
    predictions, distributions = topic_model.fit_transform(bgl_df_train["ComponentEventTemplate"], y=[int(label != "-") for label in bgl_df_train["Label"]])

    print(predictions)
    print(topic_model.get_topic_info())
    print(topic_model.get_topic_freq())

    hierarchical_topics = topic_model.hierarchical_topics(bgl_df_train["ComponentEventTemplate"])
    tree = topic_model.get_topic_tree(hierarchical_topics)
    print(tree)

    topic_shannon_entropy(predictions, bgl_df_train["Label"])


if __name__ == "__main__":
    main()



