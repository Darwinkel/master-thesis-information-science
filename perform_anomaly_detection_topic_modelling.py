import math
from collections import Counter
import random
from statistics import mean

from bertopic import BERTopic
from pandas import DataFrame

import numpy as np
import pandas as pd
from scipy.stats import entropy, hmean
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import homogeneity_score, adjusted_mutual_info_score, completeness_score


# Fix Python and numpy seeds
random.seed(42)
np.random.seed(42)

def get_topics(vectorizer, H, top_words):
    id2word = {i: k for i, k in enumerate(
        vectorizer.get_feature_names_out())}


    topic_list = []

    for topic in H:
        words_list = sorted(
            list(enumerate(topic)), key=lambda x: x[1], reverse=True)

        if top_words == -1:
            topk = [tup[0] for tup in words_list]
        else:
            topk = [tup[0] for tup in words_list[0:top_words]]

        topic_list.append([id2word[i] for i in topk])

    return topic_list

def mean_topic_entropy_score(topic_entropies: list) -> float:
    divisor = 1  # log2(2)
    if len(topic_entropies) > 1:
        divisor = math.log(len(topic_entropies), 2)

    return mean(topic_entropies) / divisor


def topic_shannon_entropy(predictions, labels):
    '''Calculate Shannon information entropy of a topic. Assumes that anomaly yes/no is binary.'''

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

        weight = freq_counts[1] / len(distribution)
        if weight != 0:
            weights.append(freq_counts[1])
            entropies.append(freq_counts[1] / len(distribution))

    print(entropies)
    print(mean_topic_entropy_score(entropies))
    return entropies, mean_topic_entropy_score(entropies)


def do_anomaly_detection(model: str, df_train: DataFrame, df_test: DataFrame, num_topics: int, labels: list[int]):
    vectorizer_model = CountVectorizer(max_df=1.0, min_df=1, ngram_range=(1, 1), strip_accents='unicode',
                                       stop_words='english', lowercase=False)

    df_train_ComponentEventTemplate = df_train["ComponentEventTemplate"].to_numpy()
    df_test_ComponentEventTemplate = df_test["ComponentEventTemplate"].to_numpy()

    if model == "LDA":
        vectorizer_model.fit(df_train_ComponentEventTemplate)
        lda = LatentDirichletAllocation(n_components=num_topics).fit(vectorizer_model.transform(df_train_ComponentEventTemplate))
        doc_topic_distr = lda.transform(vectorizer_model.transform(df_test_ComponentEventTemplate))
        predictions = np.argmax(doc_topic_distr, axis=1)
        print("Important: LDA topics")
        topics = get_topics(vectorizer_model, lda.components_, 10)
        topic_counts = Counter(predictions)
        counts = []
        for idx, topic in enumerate(topics):
            counts.append(topic_counts[idx])


    else:
        topic_model = BERTopic(verbose=True, vectorizer_model=vectorizer_model, calculate_probabilities=False,
                               top_n_words=-1, nr_topics=num_topics)

        topic_model.fit(df_train_ComponentEventTemplate, y=labels)

        predictions, distributions = topic_model.transform(df_test_ComponentEventTemplate)

        print("Important: BERTopic topics")
        counts = []
        topics = []
        for idx, row in topic_model.get_topic_info().iterrows():
            counts.append(row["Count"])
            topics.append(row["Representation"][:10])

    binary_labels = [int(label != "-") for label in df_test["Label"]]
    print(binary_labels)
    print(predictions)
    print(counts)
    print(topics)
    print({
        "homogeneity_score": homogeneity_score(binary_labels, predictions),
        "completeness_score": completeness_score(binary_labels, predictions),
        "ami_score": adjusted_mutual_info_score(binary_labels, predictions),
    })

    return counts, topics, {
        "homogeneity_score": homogeneity_score(binary_labels, predictions),
        "completeness_score": completeness_score(binary_labels, predictions),
        "ami_score": adjusted_mutual_info_score(binary_labels, predictions),
    }


def main():
    thunderbird_df_train = pd.read_csv("datasets_anomaly_detection/thunderbird_train.tsv", sep="\t")
    thunderbird_df_test = pd.read_csv("datasets_anomaly_detection/thunderbird_test.tsv", sep="\t")
    bgl_df_train = pd.read_csv("datasets_anomaly_detection/bgl_train.tsv", sep="\t")
    bgl_df_test = pd.read_csv("datasets_anomaly_detection/bgl_test.tsv", sep="\t")

    all_data = pd.concat([thunderbird_df_train, bgl_df_train, thunderbird_df_test, bgl_df_test])

    # Within dataset x2
    # Across dataset x2
    datasets = {
        "BGL-within": (bgl_df_train, bgl_df_test),
        "TB-within": (thunderbird_df_train, thunderbird_df_test),
        "BGL-TB-cross": (
            pd.concat([bgl_df_train, bgl_df_test]), pd.concat([thunderbird_df_train, thunderbird_df_test])),
        "TB-BGL-cross": (
            pd.concat([thunderbird_df_train, thunderbird_df_test]), pd.concat([bgl_df_train, bgl_df_test])),
        "BGL+TB-TB-cross": (
            pd.concat([bgl_df_train, thunderbird_df_train]), thunderbird_df_test),
        "BGL+TB-BGL-cross": (
            pd.concat([bgl_df_train, thunderbird_df_train]), bgl_df_test),
        "All": (all_data, all_data)
    }

    results = []
    for dataset_key, train_test_tuple in datasets.items():
        train_df, test_df = train_test_tuple
        for num_topics in [4, 16, 8, 16, 32]:
            # Supervised by all classes (no need to change metric; binary is easier than multiclass so its actually quite lenient)
            # Supervised anomaly yes/no
            # Unsupervised (LDA, BERTopic baselines)
            labels = {
                "Unsupervised": None,
                "Binary": [int(label != "-") for label in train_df["Label"]],
                "Multiclass": LabelEncoder().fit_transform(train_df["Label"]),
            }

            for label_key, label in labels.items():

                # Do LDA as well
                if label_key == "Unsupervised":
                    print(f"LDA - {dataset_key} - {num_topics} - {label_key}")
                    counts, topics, score = do_anomaly_detection("LDA", train_df, test_df, num_topics, label)

                    results.append({
                        "Model": "LDA",
                        "Dataset": dataset_key,
                        "No. topics": num_topics,
                        "Label type": label_key,
                        "Homogeneity Score": score["homogeneity_score"],
                        "Completeness Score": score["completeness_score"],
                        "AMI Score": score["ami_score"],
                        "Counts": counts,
                        "Topics": topics,
                    })

                print(f"BERTopic - {dataset_key} - {num_topics} - {label_key}")
                counts, topics, score = do_anomaly_detection("BERTopic", train_df, test_df, num_topics, label)

                results.append({
                    "Model": "BERTopic",
                    "Dataset": dataset_key,
                    "No. topics": num_topics,
                    "Label type": label_key,
                    "Homogeneity Score": score["homogeneity_score"],
                    "Completeness Score": score["completeness_score"],
                    "AMI Score": score["ami_score"],
                    "Counts": counts,
                    "Topics": topics,
                })

    pd.DataFrame(results).to_csv("anomaly_detection_results.tsv", sep="\t")


if __name__ == "__main__":
    main()
