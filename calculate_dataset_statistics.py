import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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

    # Class distribution

    # print("")
    # print(thunderbird_df_train["Label"].value_counts())
    # print(thunderbird_df_test.value_counts())
    # print(bgl_df_train.value_counts())
    # print(bgl_df_test.value_counts())


if __name__ == "__main__":
    main()
