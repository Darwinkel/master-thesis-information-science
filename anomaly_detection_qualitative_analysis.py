import ast
from statistics import mean

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel, kendalltau, wilcoxon
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import seaborn as sns
import json

def main():

    results_df = pd.read_csv('anomaly_detection_results_qualitative.tsv', sep="\t")

    bertopic_unique_words = set()
    lda_unique_words = set()

    df_counts = pd.DataFrame()

    for idx, row in results_df.iterrows():
        counts = ast.literal_eval(row["Counts"])
        topics = ast.literal_eval(row["Topics"])
        print(f"\n{row['Model']} - {row['No. topics']} - {row['Label type']}")
        print(f"HS: {row['Homogeneity Score']}")
        print(f"CS: {row['Completeness Score']}")
        print(f"AMI: {row['AMI Score']}")
        dict = {"Model": [row['Model']] * len(counts), "Documents assigned to topic": counts}
        df_counts = pd.concat([df_counts, pd.DataFrame(dict)], axis=0)
        for topic_id, topic in enumerate(topics):
            print(counts[topic_id], topic)
            if row['Model'] == "BERTopic":
                bertopic_unique_words.update(topic)
            else:
                lda_unique_words.update(topic)

    print(len(bertopic_unique_words))
    print(len(lda_unique_words))

    plt.figure(figsize=(8, 6), dpi=200)
    sns.boxplot(data=df_counts, y="Model", x="Documents assigned to topic")
    plt.tight_layout()
    plt.savefig(f"plots/boxplot_anomaly_detection_results_topic_counts.png")

if __name__ == "__main__":
    main()
