from pathlib import Path
import random


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Fix Python and numpy seeds
random.seed(42)
np.random.seed(42)

def load_supercomputer_df(file: Path):
    rcsv = pd.read_csv(file)[['Label', 'EventTemplate', 'Component']]

    # Parsing error: Thunderbird Components sometimes contains timestamps
    rcsv = rcsv[~rcsv['Component'].astype(str).str.startswith(("Apr", "May", "Jun", "Jul"))]

    rcsv['ComponentEventTemplate'] = rcsv['Component'] + " " + rcsv['EventTemplate']
    rcsv = rcsv.drop(columns=['EventTemplate', 'Component'])  # .drop_duplicates()
    rcsv_majority_class = rcsv.groupby(['ComponentEventTemplate', 'Label']).size().reset_index(name='Count').groupby(
        'ComponentEventTemplate').apply(lambda x: x[x['Count'] == x['Count'].max()])

    return rcsv_majority_class

def main():

    thunderbird_df = load_supercomputer_df(Path("templated_datasets/Thunderbird/Thunderbird_every20th_anomalies_concat.log_structured.csv"))
    bgl_df = load_supercomputer_df(Path("templated_datasets/BGL/BGL.log_structured.csv"))

    print(thunderbird_df)
    print(bgl_df)

    thunderbird_df_train, thunderbird_df_test = train_test_split(thunderbird_df, shuffle=True, random_state=42, test_size=0.2)
    bgl_df_train, bgl_df_test = train_test_split(bgl_df, shuffle=True, random_state=42, test_size=0.2)

    print(thunderbird_df_train)
    print(thunderbird_df_test)
    print(bgl_df_train)
    print(bgl_df_test)

    thunderbird_df_train.to_csv("thunderbird_train.tsv", index=False, sep="\t")
    thunderbird_df_test.to_csv("thunderbird_test.tsv", index=False, sep="\t")
    bgl_df_train.to_csv("bgl_train.tsv", index=False, sep="\t")
    bgl_df_test.to_csv("bgl_test.tsv", index=False, sep="\t")

if __name__ == "__main__":
    main()



