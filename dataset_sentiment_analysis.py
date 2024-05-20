
from scipy.stats import ttest_ind
from torch.nn import functional as F
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english").to("cuda")
TOKENIZER = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def binarize_label(text: str):
    return "Not anomaly" if text == "-" else "Anomaly"

def sentiment_analysis(text: str):
    tokens = TOKENIZER(text, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = MODEL(**tokens).logits
        probabilities_scores = F.softmax(logits, dim=-1)[0]

    # -1 if negative; 1.0; 0.0
    # 1 if positive; 0.0; 1.0

    return (probabilities_scores[1]-probabilities_scores[0]).item()

def main():

    thunderbird_df_train = pd.read_csv("thunderbird_train.tsv", sep="\t")
    thunderbird_df_test = pd.read_csv("thunderbird_test.tsv", sep="\t")
    bgl_df_train = pd.read_csv("bgl_train.tsv", sep="\t")
    bgl_df_test = pd.read_csv("bgl_test.tsv", sep="\t")

    thunderbird_df = pd.concat([thunderbird_df_train, thunderbird_df_test])
    bgl_df = pd.concat([bgl_df_train, bgl_df_test])

    bgl_df["Label"] = bgl_df["Label"].apply(binarize_label)
    thunderbird_df["Label"] = thunderbird_df["Label"].apply(binarize_label)

    print(sentiment_analysis("Hello henk, you're evil."))
    print(sentiment_analysis("Hello henk, you're cool."))

    # Histogram / distribution of template length
    thunderbird_df["Sentiment"] = thunderbird_df["ComponentEventTemplate"].apply(sentiment_analysis)
    print(thunderbird_df)
    plt.figure()
    sns.boxplot(data=thunderbird_df, x="Sentiment", y="Label")
    plt.savefig("datasets_boxplot_sentiment_thunderbird.png")

    bgl_df["Sentiment"] = bgl_df["ComponentEventTemplate"].apply(sentiment_analysis)
    print(bgl_df)
    plt.figure()
    sns.boxplot(data=bgl_df, x="Sentiment", y="Label")
    plt.savefig("datasets_boxplot_sentiment_bgl.png")

    thunderbird_df_anomaly = thunderbird_df[thunderbird_df["Label"] == "Anomaly"]["Sentiment"]
    thunderbird_df_not_anomaly = thunderbird_df[thunderbird_df["Label"] == "Not anomaly"]["Sentiment"]

    bgl_df_anomaly = bgl_df[bgl_df["Label"] == "Anomaly"]["Sentiment"]
    bgl_df_not_anomaly = bgl_df[bgl_df["Label"] == "Not anomaly"]["Sentiment"]

    print(thunderbird_df_anomaly.describe())
    print(thunderbird_df_not_anomaly.describe())

    print(bgl_df_anomaly.describe())
    print(bgl_df_not_anomaly.describe())

    print(ttest_ind(thunderbird_df_anomaly, thunderbird_df_not_anomaly))
    print(ttest_ind(bgl_df_anomaly, bgl_df_not_anomaly))

if __name__ == "__main__":
    main()
