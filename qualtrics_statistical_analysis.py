from statistics import mean

import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel, kendalltau, wilcoxon

import seaborn as sns

def calculate_kendall_correlation(df):
    kendall = []
    pairs_i = []
    pairs_j = []
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                kendall.append(df.iloc[i].corr(df.iloc[j], method='kendall'))
                pairs_i.extend(df.iloc[i].to_numpy())
                pairs_j.extend(df.iloc[j].to_numpy())

    return kendalltau(pairs_i, pairs_j)

def calculate_cohens_kappa(df):
    """Not Cohens K. Calculate average agreement between raters, not including chance because everything is different."""
    kendall = []

    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                arr1 = df.iloc[i].to_numpy()
                arr2 = df.iloc[j].to_numpy()

                amount_agreed = (arr1 == arr2).sum()
                cohens_k = (amount_agreed/len(arr1))

                kendall.append(cohens_k)
    return mean(kendall)

def calculate_wi_accuracy(df):
    accuracies = []
    for column in df.columns:
        true_term = column.split("|")[1].strip()
        counts = (df[column]==true_term).sum()
        accuracies.append(counts / len(df))
    return accuracies

def main():
    trial_df = pd.read_csv("survey_data/NTM Trial 1_April 3, 2024_11.03 fixed headers.tsv", sep="\t", header=0, skiprows=0, encoding="utf-16")

    valid_trial_df = trial_df.loc[trial_df['Finished'] == True]
    valid_trial_df = valid_trial_df.loc[valid_trial_df['Q99'] != "Never (less than once per year)"]
    valid_trial_df = valid_trial_df.loc[valid_trial_df['DistributionChannel'] != "preview"]

    # Quality rating
    quality_df = valid_trial_df.filter(regex='^Rating\d+$', axis='columns').replace({'Not very related': 0, 'Somewhat related': 1, 'Very related': 2})
    print(quality_df)
    print(calculate_kendall_correlation(quality_df))


    # Anomaly rating
    anomaly_df = valid_trial_df.filter(regex='^Rating\d+a$', axis='columns').replace({'Not likely to be anomalous': 0, 'Somewhat likely to be anomalous': 1, 'Very likely to be anomalous': 2})
    print(anomaly_df)
    print(calculate_kendall_correlation(anomaly_df))

    # Word intrusion
    wi_df = valid_trial_df.filter(regex='^Intrusion', axis='columns')
    wi_accuracies = calculate_wi_accuracy(wi_df)
    print(wi_df)
    print(mean(wi_accuracies))
    print(calculate_cohens_kappa(wi_df))



    ##########
    # Metric correlation #
    ##########

    metric_df = pd.read_csv("survey_data/qualtrics_out.tsv", sep="\t", header=0)
    print(metric_df)

    quality_df_mean = quality_df.mean(axis=0).reset_index(drop=True)
    anomaly_df_mean = anomaly_df.mean(axis=0).reset_index(drop=True)
    wi_accuracies_df = pd.Series(wi_accuracies)

    metrics = ["cnpmi", "ccv", "umass", "cuci", "pjs", "td", "kld"]

    for metric in metrics:
        print(metric)
        temp_df = metric_df[metric].reset_index(drop=True)
        print(kendalltau(quality_df_mean, temp_df))
        print(kendalltau(anomaly_df_mean, temp_df))
        print(kendalltau(wi_accuracies_df, temp_df))

    print(wi_accuracies_df)

    ##########
    # Comparing feature-wise #
    ##########

    metric_df["quality"] = quality_df_mean
    metric_df["anomaly"] = anomaly_df_mean
    metric_df["wi_accuracy"] = wi_accuracies_df

    print(metric_df)

    metric_df = metric_df.rename(columns={"model": "Model", "remove_digits_special_chars": "Remove digits & special characters", "lowercase": "Lowercase everything", "split_words": "Split words", "quality": "Quality rating", "anomaly": "Anomaly rating", "wi_accuracy": "Word intrusion accuracy"})

    metric_df = metric_df.replace({"Model": {"lda": "LDA", "bertopic": "BERTopic"}})

    print(metric_df)

    features = ["Model", "Remove digits & special characters", "Lowercase everything", "Split words"]

    df_melted = pd.melt(metric_df, id_vars=features, value_vars=['Quality rating', 'Anomaly rating', 'Word intrusion accuracy'], var_name="Task", value_name="Score")

    for feature in features:
        plt.figure(figsize=(8, 6), dpi=200)
        sns.boxplot(data=df_melted, y='Task', x='Score', hue=feature)
        plt.tight_layout()
        plt.savefig(f"plots/boxplot_configuration_{feature}.png")

        quality_groups = tuple(metric_df.groupby(by=feature)["Quality rating"])
        anomaly_groups = tuple(metric_df.groupby(by=feature)["Anomaly rating"])
        wi_accuracy_groups = tuple(metric_df.groupby(by=feature)["Word intrusion accuracy"])

        print(metric_df.groupby(by=feature)["Quality rating"].describe())
        print(ttest_rel(quality_groups[0][1], quality_groups[1][1]))
        print(wilcoxon(quality_groups[0][1], quality_groups[1][1]))

        print(metric_df.groupby(by=feature)["Anomaly rating"].describe())
        print(ttest_rel(anomaly_groups[0][1], anomaly_groups[1][1]))
        print(wilcoxon(anomaly_groups[0][1], anomaly_groups[1][1]))

        print(metric_df.groupby(by=feature)["Word intrusion accuracy"].describe())
        print(ttest_rel(wi_accuracy_groups[0][1], wi_accuracy_groups[1][1]))
        print(wilcoxon(wi_accuracy_groups[0][1], wi_accuracy_groups[1][1]))




if __name__ == "__main__":
    main()
