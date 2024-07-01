from pathlib import Path
import random

import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
from bertopic import BERTopic
import re
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.similarity_metrics import PairwiseJaccardSimilarity
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, KLDivergence
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


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


    no_top_words = 10
    no_top_words_intrusion = 5
    num_topics = 8

    print(f"{model_type} - parsing {parsing} - lowercase {lowercase} - remove_digits_special_chars {remove_digits_special_chars} - split_words {split_words}")

    if split_words:
        documents = documents.apply(lambda x: re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', x))

    if lowercase:
        documents = documents.apply(lambda x: x.lower())

    if remove_digits_special_chars:
        documents = documents.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
        documents = documents.apply(lambda x: re.sub(r'\d', ' ', x))

    print(documents.iloc[0])
    print(documents.iloc[10])
    print(documents.iloc[-1])

    # General note: heavy preprocessing is not really desirable in BERTopic embedding generation
    vectorizer_model = CountVectorizer(max_df=0.99, min_df=2, ngram_range=(1, 1), strip_accents='unicode', stop_words='english', lowercase=False)

    tokenized_documents = documents.apply(vectorizer_model.build_analyzer())

    # We follow some literature by filtering out documents with less than 3 words, and each word must be greater than 2
    raw_documents_without_empty = []
    for idx, doc in enumerate(tokenized_documents):
        if len(doc) > 2:
            if all(len(ele) > 2 for ele in doc):
                raw_documents_without_empty.append(documents.iloc[idx])

    print(raw_documents_without_empty[0])
    print(raw_documents_without_empty[10])
    print(raw_documents_without_empty[-1])


    if model_type == "lda":

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

    elif model_type == "bertopic":

        topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=num_topics, calculate_probabilities=True, top_n_words=-1)
        _, distributions = topic_model.fit_transform(raw_documents_without_empty)

        topics = []
        complete_topics = []
        for word_score in topic_model.get_topics().values():
            complete_topics.append([t[0] for t in word_score])
            if len(topics) < no_top_words:
                topics.append([t[0] for t in word_score])

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

    df_template_deduplicated_documents = []
    for file in Path("templated_datasets").rglob("*_templates.csv"):
        df_template_deduplicated_documents.append(pd.read_csv(file)["EventTemplate"])
    template_deduplicated_documents = pd.concat(df_template_deduplicated_documents).astype('str').drop_duplicates()

    print(template_deduplicated_documents)

    combinations = []
    for idx, documents in enumerate([template_deduplicated_documents]):
        document_preprocessing_type = ["templated_deduplicated"][idx]

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



