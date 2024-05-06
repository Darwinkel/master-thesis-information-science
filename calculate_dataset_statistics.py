import pandas as pd


def main():

    thunderbird_df_train = pd.read_csv("thunderbird_train.tsv", sep="\t")
    thunderbird_df_test = pd.read_csv("thunderbird_test.tsv", sep="\t")
    bgl_df_train = pd.read_csv("bgl_train.tsv", sep="\t")
    bgl_df_test = pd.read_csv("bgl_test.tsv", sep="\t")

    # Histogram / distribution of the templates (just the count column)

    # Histogram / distribution of the classes

    # Class distribution

    print("")
    print(thunderbird_df_train["Label"].value_counts())
    print(thunderbird_df_test.value_counts())
    print(bgl_df_train.value_counts())
    print(bgl_df_test.value_counts())


if __name__ == "__main__":
    main()
