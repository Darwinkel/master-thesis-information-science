from collections import Counter

import numpy
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.tokenize import wordpunct_tokenize

def main():

    thunderbird_df_train = pd.read_csv("thunderbird_train.tsv", sep="\t")
    thunderbird_df_test = pd.read_csv("thunderbird_test.tsv", sep="\t")
    bgl_df_train = pd.read_csv("bgl_train.tsv", sep="\t")
    bgl_df_test = pd.read_csv("bgl_test.tsv", sep="\t")

    thunderbird_df = pd.concat([thunderbird_df_train, thunderbird_df_test])
    bgl_df = pd.concat([bgl_df_train, bgl_df_test])

    # Histogram / distribution of the templates (just the count column)
    print(thunderbird_df['Label'].value_counts())
    print(len(thunderbird_df))
    plt.figure()
    plt.xticks(rotation=90)
    thunderbird_only_anomalies = thunderbird_df[thunderbird_df["Label"] != "-"]
    sns.countplot(data=thunderbird_only_anomalies, x="Label", order = thunderbird_only_anomalies['Label'].value_counts().index)
    plt.savefig(f"datasets_countplot_thunderbird.png")

    print(bgl_df['Label'].value_counts())
    print(len(bgl_df))
    plt.figure()
    plt.xticks(rotation=90)
    bgl_only_anomalies = bgl_df[bgl_df["Label"] != "-"]
    sns.countplot(data=bgl_only_anomalies, x="Label", order = bgl_only_anomalies['Label'].value_counts().index)
    plt.savefig(f"datasets_countplot_bgl.png")

    # Histogram / distribution of template length
    print(thunderbird_df["ComponentEventTemplate"].apply(len).describe())
    thunderbird_df["Input length"] = thunderbird_df["ComponentEventTemplate"].apply(len)
    plt.figure()
    sns.boxplot(data=thunderbird_df, x="Input length", y="Label")
    plt.savefig("datasets_boxplot_thunderbird.png")

    print(bgl_df["ComponentEventTemplate"].apply(len).describe())
    bgl_df["Input length"] = bgl_df["ComponentEventTemplate"].apply(len)
    plt.figure()
    sns.boxplot(data=bgl_df, x="Input length", y="Label")
    plt.savefig("datasets_boxplot_bgl.png")


    # Vocab distribution in datasets
    thunderbird_vocab = Counter()
    thunderbird_df['ComponentEventTemplate'].str.lower().apply(wordpunct_tokenize).apply(thunderbird_vocab.update)
    #print(pd.DataFrame.from_dict(thunderbird_vocab, orient='index').reset_index())

    sorted_word_counts = thunderbird_vocab.most_common()

    # Extract frequencies
    frequencies = [count for word, count in sorted_word_counts]

    # Create rank array (1, 2, 3, ...)
    ranks = list(range(1, len(frequencies) + 1))

    # Plot using seaborn
    plt.figure()
    sns.lineplot(x=ranks, y=frequencies)

    # Calculate the ideal Zipfian distribution
    C = frequencies[0]  # C is the frequency of the most common word
    s = 1  # Zipf's exponent, often close to 1

    # Generate the ideal Zipfian frequencies
    ideal_frequencies = [C / (r ** s) for r in ranks]

    # Plot the ideal Zipfian distribution
    plt.plot(ranks, ideal_frequencies, color='red', marker='o', linestyle='dashed', linewidth=0.2, markersize=0.2,
             label='Ideal Zipfian Distribution')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Zipfian Distribution of Word Frequencies')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("datasets_vocab_histogram_thunderbird.png")
    #exit()

    # plt.figure()
    # sns.histplot(data=pd.DataFrame.from_dict(thunderbird_vocab, orient='index').reset_index(), binwidth=2)
    # plt.savefig("datasets_vocab_histogram_thunderbird.png")

    bgl_vocab = Counter()
    bgl_df['ComponentEventTemplate'].str.lower().apply(wordpunct_tokenize).apply(bgl_vocab.update)
    # plt.figure()
    # sns.histplot(data=pd.DataFrame.from_dict(bgl_vocab, orient='index').reset_index())
    # plt.savefig("datasets_vocab_histogram_bgl.png")

    sorted_word_counts = bgl_vocab.most_common()

    # Extract frequencies
    frequencies = [count for word, count in sorted_word_counts]

    # Create rank array (1, 2, 3, ...)
    ranks = list(range(1, len(frequencies) + 1))

    # Plot using seaborn
    plt.figure()
    sns.lineplot(x=ranks, y=frequencies)

    # Calculate the ideal Zipfian distribution
    C = frequencies[0]  # C is the frequency of the most common word
    s = 1  # Zipf's exponent, often close to 1

    # Generate the ideal Zipfian frequencies
    ideal_frequencies = [C / (r ** s) for r in ranks]

    # Plot the ideal Zipfian distribution
    plt.plot(ranks, ideal_frequencies, color='red', marker='o', linestyle='dashed', linewidth=0.2, markersize=0.2,
             label='Ideal Zipfian Distribution')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Zipfian Distribution of Word Frequencies')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("datasets_vocab_histogram_bgl.png")

    print(thunderbird_vocab)
    print(bgl_vocab)

    # Class distribution

    # print("")
    # print(thunderbird_df_train["Label"].value_counts())
    # print(thunderbird_df_test.value_counts())
    # print(bgl_df_train.value_counts())
    # print(bgl_df_test.value_counts())


if __name__ == "__main__":
    main()
