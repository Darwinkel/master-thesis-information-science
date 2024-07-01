# Human-interpretable Topic Models for Explainable Anomaly Detection in Multidomain System Log Analysis (Darwinkel, 2024)

Code and datasets for my master's thesis. The code is messy, but considering nobody will probably try to replicate this study, that doesn't matter too much. Contact me if you're interested and need help setting things up.

# License
The code of this project is licensed under the GNU GPLv3. The data that I have collected and processed (e.g. in the `survey_data` folder) is licensed under the Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.

# Installation and dependencies

- See `pyproject.toml` and `poetry.lock` for the list of dependencies and their versions
- Note that [OCTIS](https://github.com/MIND-Lab/OCTIS) currently requires manual installation because of a dependency on old scikit-learn. This has already been reported in the upstream OCTIS repository, but hasn't been fixed yet.
- A slightly modified version of [SwissLog](https://github.com/IntelligentDDS/SwissLog) is contained in the `Adapted-SwissLog` folder.

# Scripts / functionality

### `Adapted-SwissLog/run.py`
Adapted version of SwissLog which parses [LogHub](https://github.com/logpai/loghub) datasets. The output data is needed for the other scripts.

### `prepare_qualtrics_experiment.py`

Creates Qualtrics survey from `templated_datasets/*`.

### `qualtrics_statistical_analysis.py`

Runs statistical tests and creates plots from `survey_data/qualtrics_out.tsv`.

### `create_anomaly_detection_datasets.py`

Creates usable BGL and Thunderbird datasets from `templated_datasets/*`.

### `dataset_sentiment_analysis.py`

Runs and plots sentiment analysis on `datasets_anomaly_detection/*`.

### `dataset_statistical_analysis.py`

Prints and plots statistics on `datasets_anomaly_detection/*`.

### `perform_anomaly_detection_topic_modelling.py`

Runs topic modelling based anomaly detection on `datasets_anomaly_detection/*`.

### `anomaly_detection_quantitative_analysis.py`

Creates a bunch of plots from `results_anomaly_detection/anomaly_detection_results.tsv`.

### `anomaly_detection_qualitative_analysis.py`

Prints some statistics and plots the amount of documents assigned to topic on `results_anomaly_detection/anomaly_detection_results_qualitative.tsv`.
