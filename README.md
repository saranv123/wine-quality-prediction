# Wine Quality Prediction: Classification and Regression

A machine learning project exploring the prediction of white-wine quality scores from physicochemical measurements. The notebook compares classification and regression approaches using scikit-learn.

## Project objective

The goal is to predict the sensory quality score of white wine using its physicochemical properties. The target, `quality`, is an integer score. Two approaches are investigated:

- **Classification:** predict the quality category directly.
- **Regression:** predict a continuous quality score, then round it to the nearest integer for comparison with the classification models.

The project uses Logistic Regression, Random Forest, Support Vector Machine (RBF kernel), Gradient Boosting, Ridge Regression, K-Nearest Neighbors Regression, and Random Forest Regression.

## Dataset

- **Source:** [UCI Machine Learning Repository – Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Dataset used:** White wine
- **Observations:** 4,898
- **Input features:** 11 physicochemical measurements
- **Target:** wine quality score

Download the UCI Wine Quality dataset and extract `winequality-white.csv` into the `wine+quality/` directory. The notebook reads the file using:

```python
pd.read_csv('wine+quality/winequality-white.csv', sep=';')
```

The dataset itself is not included in this repository.

## Repository structure

```text
wine-quality-prediction/
├── notebooks/
│   └── wine_quality_analysis.ipynb
├── wine+quality/
│   └── README.md
├── .gitignore
├── README.md
└── requirements.txt
```

## Methodology

1. Load the white-wine dataset and select the 11 physicochemical features.
2. Separate the features (`X`) and target (`y`).
3. Split the observations into training and test sets using an 80/20 split, with stratification and `random_state=0`.
4. Define a shared numerical preprocessing pipeline using median imputation and `StandardScaler`.
5. Use four-fold `StratifiedKFold` cross-validation (`shuffle=True`, `random_state=0`) for hyperparameter tuning.
6. Use `GridSearchCV` to select model hyperparameters.
7. Evaluate the selected models on the held-out test set.

The notebook retains the original model-specific functions, global preprocessing pipeline, cross-validation object, and evaluation functions.

## Models

### Classification

- Logistic Regression
- Support Vector Machine with an RBF kernel
- Random Forest Classifier
- Gradient Boosting Classifier

The classification grid searches use **accuracy** as the scoring metric.

### Regression

- Ridge Regression
- K-Nearest Neighbors Regression
- Random Forest Regression

The regression grid searches use **\(R^2\)** as the scoring metric. Continuous predictions are rounded to the nearest integer and clipped to the quality-score range used in the notebook before calculating accuracy and quadratic weighted kappa.

## Evaluation metrics

- **Accuracy:** fraction of predictions that exactly match the true quality score.
- **Quadratic weighted kappa (QWK):** measures agreement while penalizing larger disagreements more heavily than smaller disagreements.
- **\(R^2\):** measures the proportion of target variance explained by the regression predictions.

QWK is particularly informative for this problem because quality scores are ordered. A prediction one score away from the true value is a less severe disagreement than a prediction several scores away.

## Results

The following values are the results recorded in the original notebook.

| Model | Accuracy | Quadratic weighted kappa | \(R^2\) |
|---|---:|---:|---:|
| Logistic Regression | 0.54 | 0.43 | — |
| Random Forest Classifier | 0.66 | 0.65 | — |
| SVM (RBF) | 0.67 | 0.56 | — |
| Gradient Boosting Classifier | 0.60 | 0.56 | — |
| Ridge Regression | 0.53 | 0.41 | 0.31 |
| KNN Regression | 0.54 | 0.49 | 0.39 |
| Random Forest Regression | **0.67** | **0.66** | **0.53** |

The notebook contains the individual test-set evaluation outputs for all models. Random Forest Regression was the best-performing model in the original experiment according to the reported combination of metrics.

## Baseline comparison

A `DummyClassifier` using the most-frequent strategy and a `DummyRegressor` using the mean strategy are included as simple benchmarks.

## Notes and limitations

- The quality classes are imbalanced. Performance should therefore be interpreted using more than accuracy alone.
- The classification searches optimize accuracy, while the regression searches optimize \(R^2\). QWK is used as an additional evaluation metric, not as the grid-search objective.
- Multiple models are evaluated on the same held-out test set. Since the best model is identified after comparing their test results, the final comparison is exploratory and may be optimistic. A future version could use a separate validation set or nested cross-validation for model selection.
- Rounding regression predictions makes it possible to compare them with discrete quality labels, but it discards information contained in the continuous predictions.
- The results reflect this dataset, split, parameter grids, and modelling choices; they should not be interpreted as a general measure of wine quality.

## Running the notebook

Python 3.10 or newer is recommended.

Download and extract the UCI dataset into `wine+quality/`, then launch Jupyter:

```bash
jupyter lab
```

Open `notebooks/wine_quality_analysis.ipynb` and run the cells from top to bottom.

## Acknowledgement

The dataset is provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality). Please consult the UCI page for the dataset citation and licensing information.
