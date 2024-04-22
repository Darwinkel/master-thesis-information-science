from pathlib import Path
import random
from typing import List

import numpy as np
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.similarity_metrics import PairwiseJaccardSimilarity
from sklearn.cluster import AgglomerativeClustering
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, KLDivergence
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
from bertopic import BERTopic
import spacy
import re

# Fix Python and numpy seeds
random.seed(42)
np.random.seed(42)

#nlp = spacy.load("en_core_web_sm")
# def lemmatize_string(doc):
#     tokenized_doc = []
#     for token in nlp(doc):
#         # if not token.is_stop and not token.is_space and not token.is_digit and not token.is_punct:
#         #         tokenized_doc.append(token.lemma_.lower())
#         tokenized_doc.append(token.lemma_)
#     # print(doc)
#     # print(tokenized_doc)
#     return " ".join(tokenized_doc)

# def display_topics(model, feature_names, no_top_words):
#     for topic_idx, topic in enumerate(model.components_):
#         print ("Topic %d:" % (topic_idx))
#         print ("|".join([feature_names[i]
#                         for i in topic.argsort()[:-no_top_words - 1:-1]]))
#

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

def word_intrusion(topic: list[str], other_topics: list[list[str]], top_words, top_word_multiplier=2) -> str:

    # Note: if top_n or vocab is very small, may not be able to find a word that is not in the top_n of any other topic
    for low_prob_word in reversed(topic):

        # Skip empty strings
        if low_prob_word in ("", " "):
            continue

        top_n_other_topic_words = set()
        for other_idx, other_topic in enumerate(other_topics):
            top_n_other_topic_words.update(set(other_topic[0:top_words*top_word_multiplier]))

        if low_prob_word in top_n_other_topic_words:
            if low_prob_word not in topic[:top_words]:
                print(f"Topic {topic[:top_words]} - Intruder: {low_prob_word}")
                return low_prob_word
            else:
                print(f"Word {low_prob_word} already in topic")
        else:
            print(f"Word {low_prob_word} not in [0:{top_words*top_word_multiplier}] of other topics")

    if top_word_multiplier > 990:
        print("Could not find intruder, picking random word from other topics...")
        return random.choice(tuple(top_n_other_topic_words))

    print(f"No intruder found! Increasing search to {top_word_multiplier+1}")
    return word_intrusion(topic, other_topics, top_words, top_word_multiplier=top_word_multiplier+1)

        # Iterate over the lowest probability words in the topic in reverse
        # Grab top 20 words of all other topics (set), not counting the topn displayed
        # Check if word is in the set
        # If yes, add to topic if not yet in intruder list
        # If no, continue

#     Chang 2009 (supported by Hoyle and Tea leaves):
# - Select top n words for a topic
# - Add a random word which has a low probability of occuring in this topic, but a high probability of occuring in another one
# - Shuffle
#


def write_intruders_to_qualtrics(topics: list[list[str]], intruders: list[str]) -> None:

    with open('qualtrics_out.txt', 'w') as f:
        f.write('[[AdvancedFormat]]\n')

        for idx, topic in enumerate(topics):
            f.write('[[Question:MC:SingleAnswer:Vertical]]\n')
            f.write(f'[[ID:Intrusion{idx} | {intruders[idx]}]]\n')
            f.write('Select which term is the least related to all other terms\n') #  <span style="font-family:Courier New,Courier,monospace;">{topic}</span>
            f.write('[[Choices]]\n')
            for word in topic:
                f.write(f'<span style="font-family:Courier New,Courier,monospace;">{word}</span>\n')

def write_rating_to_qualtrics(topics: list[list[str]]) -> None:

    with open('qualtrics_out_rating.txt', 'w') as f:
        f.write('[[AdvancedFormat]]\n')

        ### BIG QUESTION:
        # For the rating task, just sample a single topic, or show them all?

        for idx, topic in enumerate(topics):
            f.write('[[Question:MC:SingleAnswer:Horizontal]]\n')
            f.write(f'[[ID:Rating{idx}]]\n')
            f.write(f'Please rate how related the following terms are to each other: '
                    f'</br> <span style="font-family:Courier New,Courier,monospace;">{topic}</span>\n')
            f.write('[[Choices]]\n'
                    'Not very related\n'
                    'Somewhat related\n'
                    'Very related\n')

            f.write('[[Question:MC:SingleAnswer:Horizontal]]\n')
            f.write(f'[[ID:Rating{idx}a]]\n')
            f.write(f'Do the previously mentioned terms refer to something that should not normally happen, and perhaps trigger an alert?\n')
            f.write('[[Choices]]\n'
                    'Not likely to be anomalous\n'
                    'Somewhat likely to be anomalous\n'
                    'Very likely to be anomalous\n')


def dump_topics(documents, parsing, model_type, lowercase=False, remove_digits_special_chars=False, split_words=False):

    # NOTE: should fix seeds

    no_top_words = 10
    no_top_words_intrusion = 5
    num_topics = 8

    # if len(documents) > 2000:
    #     documents = documents.sample(2000, random_state=42)

    print(f"{model_type} - parsing {parsing} - lowercase {lowercase} - remove_digits_special_chars {remove_digits_special_chars} - split_words {split_words}")

    if split_words:
        documents = documents.apply(lambda x: re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', x))

    if lowercase:
        documents = documents.apply(lambda x: x.lower())

    # if remove_special_chars:
    #     documents = documents.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
    #
    # if remove_digits:
    #     documents = documents.apply(lambda x: re.sub(r'\d', ' ', x))

    if remove_digits_special_chars:
        documents = documents.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
        documents = documents.apply(lambda x: re.sub(r'\d', ' ', x))

    # if lemmatize:
    #     documents = documents.apply(lambda x: lemmatize_string(x))

    print(documents.iloc[0])
    print(documents.iloc[10])
    print(documents.iloc[-1])




  # :return model_output: a dictionary containing up to 4 keys: *topics*, *topic-word-matrix*,
  #       *topic-document-matrix*, *test-topic-document-matrix*. *topics* is the list of the most significant words for
  #       each topic (list of lists of strings). *topic-word-matrix* is the matrix (num topics x ||vocabulary||)
  #       containing  the probabilities of a word in a given topic. *topic-document-matrix* is the matrix (||topics|| x
  #       ||training documents||) containing the probabilities of the topics in a given training document.
  #       *test-topic-document-matrix* is the matrix (||topics|| x ||testing documents||) containing the probabilities
  #       of the topics in a given testing document.

    # result: dictionary
    # with up to 3 entries,
    # 'topics', 'topic-word-matrix' and
    # 'topic-document-matrix'

    # result = {}
    #
    # result["topic-word-matrix"] = self.trained_model.get_topics()
    # if top_words > 0:
    #     topics_output = []
    #     for topic in result["topic-word-matrix"]:
    #         top_k = np.argsort(topic)[-top_words:]
    #         top_k_words = list(reversed([self.id2word[i] for i in top_k]))
    #         topics_output.append(top_k_words)
    #     result["topics"] = topics_output
    #
    # result["topic-document-matrix"] = self._get_topic_document_matrix()

                # test_document_topic_matrix = []
                # for document in new_corpus:
                #     document_topics_tuples = self.trained_model[document]
                #     document_topics = np.zeros(
                #         self.hyperparameters["num_topics"])
                #     for single_tuple in document_topics_tuples:
                #         document_topics[single_tuple[0]] = single_tuple[1]
                #
                #     test_document_topic_matrix.append(document_topics)
                # result["test-topic-document-matrix"] = np.array(
                #     test_document_topic_matrix).transpose()


    # https://github.com/MIND-Lab/OCTIS/blob/08b52866672db0c1b61a99fc53835784c20d2492/octis/models/NMF_scikit.py#L152
    # scikit to results

    # General note: heavy preprocessing is not really desirable in BERTopic embedding generation
    vectorizer_model = CountVectorizer(max_df=0.99, min_df=2, ngram_range=(1, 1), strip_accents='unicode', stop_words='english', lowercase=False)

    # What do if documents are empty?
    #vectorizer_model = CountVectorizer(ngram_range=(1, 1), strip_accents='unicode')

    tokenized_documents = documents.apply(vectorizer_model.build_analyzer())

    # # filter out records with empty strings in the 'city' column
    # tokenized_documents_filtered = tokenized_documents.where(tokenized_documents.strip() != "")
    #
    # print(len(tokenized_documents))
    # print(len(tokenized_documents_filtered))

    # We follow some literature by filtering out documents with less than 3 words, and each word must be greater than 2. Damnit which one was it?
    raw_documents_without_empty = []
    for idx, doc in enumerate(tokenized_documents):
        if len(doc) > 2:
            if all(len(ele) > 2 for ele in doc):
                # print(doc)
                # print(orig_documents.iloc[idx])
                # print(documents.iloc[idx])
                raw_documents_without_empty.append(documents.iloc[idx])

    #exit()
    # print(raw_documents_without_empty)

    # filtered_series = tokenized_documents[tokenized_documents.str.len() > 1]
    # print(filtered_series)

    print(raw_documents_without_empty[0])
    print(raw_documents_without_empty[10])
    print(raw_documents_without_empty[-1])


    if model_type == "lda":
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model


        # tf_feature_names = tf_vectorizer.get_feature_names_out()

        # print(tf)
        # print(tf_feature_names)

        tf = vectorizer_model.fit_transform(raw_documents_without_empty)
        # Run LDA
        lda = LatentDirichletAllocation(n_components=num_topics).fit(tf)

        H = lda.components_
        complete_topics = get_topics(vectorizer_model, H, -1)

        random_topic_indices = random.sample(range(1, len(complete_topics)), 2)
        selected_intruder_topic = complete_topics[random_topic_indices[0]]
        selected_rating_topic = complete_topics[random_topic_indices[1]][:no_top_words]
        intruder_word = word_intrusion(selected_intruder_topic, complete_topics[:random_topic_indices[0]] + complete_topics[random_topic_indices[0]+1 :], no_top_words)

        intruded_topic = selected_intruder_topic[:no_top_words_intrusion]
        intruded_topic.append(intruder_word)
        random.shuffle(intruded_topic)



        result = {"topic-word-matrix": H,
                  "topics": np.array(get_topics(vectorizer_model, H, no_top_words)),
                  "topic-document-matrix": np.array(lda.transform(tf)).transpose(),
                  "random_word_intrusion_topic": intruded_topic,
                  "random_word_intrusion_word": intruder_word,
                  "random_rating_topic": selected_rating_topic
                  }


        # words = tf_vectorizer.get_feature_names_out()
        # print(words)

        # print(result["topic-word-matrix"].shape)
        # #print(result["topic-word-matrix"])
        # print(result["topics"].shape)
        # #print(result["topics"])
        # print(result["topic-document-matrix"].shape)
        # #print(result["topic-document-matrix"])



    # elif model_type == "nmf":
    #     # NMF is able to use tf-idf
    #     tfidf_vectorizer = TfidfVectorizer(max_df=0.99, min_df=2, stop_words='english', strip_accents="unicode", ngram_range=(1, 1))
    #     tfidf = tfidf_vectorizer.fit_transform(raw_documents_without_empty)
    #     tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    #
    #     # Run NMF
    #     nmf = NMF(n_components=num_topics).fit(tfidf)
    #     display_topics(nmf, tfidf_feature_names, no_top_words)


    elif model_type == "bertopic":
        # vectorizer_model = CountVectorizer(min_df=2, analyzer=lemmatize)

        topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=num_topics, calculate_probabilities=True, top_n_words=-1)
        _, distributions = topic_model.fit_transform(raw_documents_without_empty)


        # hierarchical_topics = topic_model.hierarchical_topics(documents)
        # tree = topic_model.get_topic_tree(hierarchical_topics)
        # print(tree)


        #print(topics)

        #print(topic_model.get_topic_freq())
        #print(topic_model.get_document_info(documents[0:10]))

        # We cannot use probabilities and distributions directly from bertopic since we use agglomerative clustering
        # topic_distr, _ = topic_model.approximate_distribution(documents, calculate_tokens=False)
        #
        # print(topics)
        # print(distributions.shape)
        # print(distributions)
        # print(topic_distr.shape)
        # print(topic_distr)
        # exit()


        #topics = topic_model.get_topics() # dict["0"] = [tuple[word, score]], topics
        #print(topic_distr.shape) # topic-document-matrix, number of topics x number of documents in the corpus
        #print(topic_token_distr.shape) # topic-word-matrix, number of topics x vocab length
        #exit()
        #print(topic_model.probabilities_)
        #print(topic_model.topics_)
        #print(topic_model.topic_sizes_)


        topics = []
        complete_topics = []
        for word_score in topic_model.get_topics().values():
            complete_topics.append([t[0] for t in word_score])
            if len(topics) < no_top_words:
                topics.append([t[0] for t in word_score])

        # print(topic_model.get_topics())
        # print(complete_topics)

        random_topic_indices = random.sample(range(1, len(complete_topics)), 2)
        selected_intruder_topic = complete_topics[random_topic_indices[0]]
        selected_rating_topic = complete_topics[random_topic_indices[1]][:no_top_words]
        intruder_word = word_intrusion(selected_intruder_topic, complete_topics[:random_topic_indices[0]] + complete_topics[random_topic_indices[0]+1 :], no_top_words)

        intruded_topic = selected_intruder_topic[:no_top_words_intrusion]
        intruded_topic.append(intruder_word)
        random.shuffle(intruded_topic)


        # For BERTopic, we discard the outlier topic for intruder and random rating
        result = {"topic-word-matrix": topic_model.c_tf_idf_.toarray(),
                  "topics": np.array(complete_topics),
                  "topic-document-matrix": distributions.transpose(),
                "random_word_intrusion_topic": intruded_topic,
                "random_word_intrusion_word": intruder_word,
                "random_rating_topic": selected_rating_topic}


    td = TopicDiversity(topk=no_top_words)  # Initialize metric
    pjs = PairwiseJaccardSimilarity(topk=no_top_words)
    kld = KLDivergence()
    ccv = Coherence(measure='c_v', topk=no_top_words, texts=tokenized_documents)
    cnpmi = Coherence(measure='c_npmi', topk=no_top_words, texts=tokenized_documents)
    umass = Coherence(measure='u_mass', topk=no_top_words, texts=tokenized_documents)
    cuci = Coherence(measure='c_uci', topk=no_top_words, texts=tokenized_documents)


    result.update({
        "td": td.score(result),
        "pjs": pjs.score(result),
        "kld": kld.score(result),
        "ccv": ccv.score(result),
        "cnpmi": cnpmi.score(result),
        "umass": umass.score(result),
        "cuci": cuci.score(result),
    })

    return result



def main():

    # df_template_documents = []
    # for file in Path("LogParserResult").rglob("*_structured.csv"):
    #     df_template_documents.append(pd.read_csv(file))
    # template_documents = pd.concat(df_template_documents)["EventTemplate"]
    #
    # print(template_documents)

    # Level,Node,Component,EventTemplate

    # df_template_deduplicated_documents = []
    # for file in Path("templated_datasets").rglob("*_structured.csv"):
    #     # if "Zookeeper" in file.stem:
    #     #     df_template_deduplicated_documents.append(pd.read_csv(file).loc[:, ['Level', "EventTemplate"]])
    #     df_template_deduplicated_documents.append(pd.read_csv(file)["EventTemplate"])
    # template_deduplicated_documents = pd.concat(df_template_deduplicated_documents).drop_duplicates()
    #
    # print(template_deduplicated_documents)

    df_template_deduplicated_documents = []
    for file in Path("templated_datasets").rglob("*_templates.csv"):
        # if "Zookeeper" in file.stem:
        #     df_template_deduplicated_documents.append(pd.read_csv(file).loc[:, ['Level', "EventTemplate"]])
        df_template_deduplicated_documents.append(pd.read_csv(file)["EventTemplate"])
    template_deduplicated_documents = pd.concat(df_template_deduplicated_documents).astype('str').drop_duplicates()

    print(template_deduplicated_documents)

    # for row in template_deduplicated_documents:
    #     print(row)
    #
    # exit()
    #
    # df_template_deduplicated_documents = []
    # for file in Path("LogParserResult").rglob("*_structured.csv"):
    #     df_template_deduplicated_documents.append(pd.read_csv(file))
    # template_deduplicated_documents = pd.concat(df_template_deduplicated_documents)["EventTemplate"].drop_duplicates()
    #
    # print(template_deduplicated_documents)
    #
    # df_raw_documents = []
    # for file in Path("../logs").rglob("*_structured.csv"):
    #     df_raw_documents.append(pd.read_csv(file))
    # raw_documents = pd.concat(df_raw_documents)["Content"].drop_duplicates()
    #
    # print(raw_documents)


    combinations = []
    for idx, documents in enumerate([template_deduplicated_documents]):
        document_preprocessing_type = ["templated_deduplicated"][idx]

        # if document_preprocessing_type != "templated_deduplicated":
        #     continue

        for remove_digits_special_chars in [True, False]:
            for lowercase in [True, False]:
                for split_words in [True, False]:
                    for model in ["lda", "bertopic"]:
                        result_dict = dump_topics(documents, document_preprocessing_type, model, lowercase, remove_digits_special_chars, split_words)
                        if result_dict is not None:

                            combinations.append({
                                "model": model,
                                "templating": document_preprocessing_type,
                                "remove_digits_special_chars": remove_digits_special_chars,
                                "lowercase": lowercase,
                                "split_words": split_words,
                                # "topics": result_dict["topics"],
                                # "topic-word-matrix": result_dict["topic-word-matrix"],
                                # "topic-document-matrix": result_dict["topic-document-matrix"],
                                "td": result_dict["td"],
                                "pjs": result_dict["pjs"],
                                "kld": result_dict["kld"],
                                "ccv": result_dict["ccv"],
                                "cnpmi": result_dict["cnpmi"],
                                "umass": result_dict["umass"],
                                "cuci": result_dict["cuci"],
                                "random_rating_topic": list(filter(None, result_dict["random_rating_topic"])),
                                "random_word_intrusion_topic": list(filter(None, result_dict["random_word_intrusion_topic"])),
                                "random_word_intrusion_word": result_dict["random_word_intrusion_word"],
                            })

    df = pd.DataFrame(combinations)

    write_intruders_to_qualtrics(df["random_word_intrusion_topic"], df["random_word_intrusion_word"])
    write_rating_to_qualtrics(df["random_rating_topic"])

    df.to_csv("qualtrics_out.tsv", sep="\t")


if __name__ == "__main__":
    main()



