# Workflow for testing MLOps platforms


## General

### Data preparation

Some basic data preparation should be possible on most platforms.
For every dataset there is a corresponding short python script mainly relying on pandas producing the
prescribed output. All datasets are publicly available in a public Amazon S3 bucket.

Ideally the following features should be present: (!!!!!!!!!!!!!!!!!!)

- Connection to different cloud and storage platforms (AWS, Google Cloud, ...)
- Versioning of the datasets after each change and linking this to a step in the model training
- Construct ETL pipelines to reproduce data processing for new data

## Classification (Pokemon data)

### Data preparation & Visualization

- Input file: pokemon.csv
- Target variable: Legendary (bool)
- Publicly hosted S3 object at: https://test-bucket2021-02-26.s3.eu-west-3.amazonaws.com/pokemon.csv (!!!!!!!!!!!!!!!!!!)
- Suggested preprocessing:
    - Remove columns: "#", "Name", "Generation"
    - Remove the two categories in "Type 1" with the least appearances (Flying: n=4; Fairy: n=17)
    - Basic data transformations like log(HP) or sqrt(HP)
    - Explore univariate histograms of numerical data and bar plots of categorical data
    - [Optional] Standardize/Normalize continuous variables. Some MLOps platforms make the conscious \
        decision not to include this in data processing, as they do this as part of their own feature \
        engineering when fitting models.
    - [Optional] One-Hot Encode categorical features. Not necessary for the same reason as before.
    - Divide data into train / test samples
- Programs:
    - 1prepare_data.py: Prepare and visualize the Pokemon dataset for classification

### Modeling & Evaluation
- Input file: Processed/pokemon-processed.csv
- Models:
    - Logistic regression (scikit-learn)
    - Support Vector Machines (scikit-learn)
    - K-Nearest Neighbors (scikit-learn)
    - Boosted decision tree (xgboost)
    - Lightgbm (lightgbm)
- Programs:
    - 2fit_models.py: Fit the models described above
- Evaluation (sorted by AUC):

| Model    |   Accuracy |        AUC |       F1 |   LogLoss |   Recall |   Precision |
|:---------|-----------:|-----------:|---------:|----------:|---------:|------------:|
| KNN      |   0.933868 |   0.857256 | 0.297872 |   2.28418 | 0.777778 |    0.184211 |
| XGB      |   0.945892 |   0.825679 | 0.597015 |   1.86886 | 0.689655 |    0.526316 |
| LR       |   0.941884 |   0.824717 | 0.52459  |   2.0073  | 0.695652 |    0.421053 |
| lightgbm |   0.93988  |   0.800197 | 0.545455 |   2.07651 | 0.642857 |    0.473684 |
| SVM      |   0.923848 | nan        | 0        | nan       | 0        |    0        |


### Deployment
- Model should monitor:
    - Fairness
    - Data drift
    - Enable accuracy evaluation (maybe months later)
- Programs:
    - 3generate_drift.py: Generate drift values for every column



## Regression (Boston housing data)

## Time series (Passenger data)

## Big data

## TODO

- Dataset with missing values
- Joining datasets from different sources