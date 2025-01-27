from statistics import mean

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel, kendalltau, wilcoxon
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import seaborn as sns

def main():

    results_df = pd.read_csv('results_anomaly_detection/anomaly_detection_results.tsv', sep="\t")

    results_df_no_lda = results_df.loc[results_df['Model'] != "LDA"]
    results_df_only_unsupervised = results_df.loc[results_df['Label type'] == "Unsupervised"]

    features = ["Dataset", "No. topics", "Label type"]

    for metric in ["Completeness", "Homogeneity", "AMI"]:
        for feature in features:
            plt.figure(figsize=(8, 6), dpi=200)
            #plt.ylim(0.0, 0.16)
            if feature == "No. topics":
                sns.boxplot(data=results_df_no_lda, x=feature, y=f"{metric} Score")
            else:
                sns.boxplot(data=results_df_no_lda, y=feature, x=f"{metric} Score")
            plt.tight_layout()
            plt.savefig(f"plots/boxplot_anomaly_detection_results_{metric}_{feature}.png")

        plt.figure(figsize=(8, 6), dpi=200)
        #plt.ylim(0.0, 0.16)
        sns.boxplot(data=results_df_only_unsupervised, y="Model", x=f"{metric} Score")
        plt.tight_layout()
        plt.savefig(f"plots/boxplot_anomaly_detection_results_lda_v_bertopic_{metric}.png")



if __name__ == "__main__":
    main()
