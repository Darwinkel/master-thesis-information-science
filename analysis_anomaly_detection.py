from statistics import mean

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel, kendalltau, wilcoxon
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import seaborn as sns

def main():


    results_df = pd.read_csv('anomaly_detection_results.tsv', sep="\t")

    results_df_no_lda = results_df.loc[results_df['Model'] != "LDA"]
    results_df_only_unsupervised = results_df.loc[results_df['Label type'] == "Unsupervised"]

    features = ["Dataset", "No. topics", "Label type"]
    for feature in features:
        plt.figure()
        sns.boxplot(data=results_df_no_lda, x=feature, y='Score')
        plt.savefig(f"boxplot_anomaly_detection_results_{feature}.png")

    plt.figure()
    sns.boxplot(data=results_df_only_unsupervised, x="Model", y='Score')
    plt.savefig(f"boxplot_anomaly_detection_results_lda_v_bertopic.png")



if __name__ == "__main__":
    main()
