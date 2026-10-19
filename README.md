# Credit Approval Prediction using Boosting

A minor project that predicts whether a credit card application will be **approved or rejected**, and compares four boosting algorithms: XGBoost, CatBoost, AdaBoost and LightGBM.

## Problem

Banks must decide quickly whether to approve or reject each credit card application. Approving a bad applicant causes a financial loss, and rejecting a good one loses a customer. This project builds a classification model to support that decision.

## Dataset

- **Name:** Credit Approval
- **Source:** [UCI Machine Learning Repository (id 27)](https://archive.ics.uci.edu/dataset/27/credit+approval)
- **Size:** 690 applications, 15 input attributes (A1 to A15) and 1 target (A16: + approved, - rejected)

## Approach

1. **Data quality check:** duplicates, missing values and unusual values.
2. **Cleaning:** missing values filled with the median (numeric) or mode (categorical); outliers capped using the IQR rule; rows were kept because the dataset is small.
3. **EDA:** four visualizations with observations (target distribution, numeric distributions, categorical vs approval, numeric vs approval).
4. **Preparation:** one-hot encoding and a stratified 80:20 train/test split. No scaling, because tree-based models do not need it. No SMOTE, because the classes are only mildly imbalanced (about 44% / 56%).
5. **Model selection:** the four boosting models were compared with 5-fold cross-validation on the training data only.
6. **Tuning:** grid search on the best model, kept only if it beat the default in cross-validation.
7. **Evaluation:** the final model was tested once on the held-out test set.

## Limitations

- Small dataset (690 rows), so scores can change with a different split.
- Hidden column meanings limit how well the results can be explained.
- Missing values and outliers were treated before the train/test split.

## Future improvements

- Use a pipeline so cleaning is fitted on training data only.
- Try a wider hyperparameter search and repeated cross-validation.
- Adjust the decision threshold to reduce false positives.

## How to run

```bash
pip install ucimlrepo xgboost catboost lightgbm scikit-learn pandas numpy matplotlib seaborn
jupyter notebook Credit_Approval_Boosting_trimmed.ipynb
```
## collab link -https://colab.research.google.com/drive/1MeLx-YDbhPQOl0p106mVrMqXKmRpdj1E?usp=sharing
The dataset is downloaded automatically with the `ucimlrepo` package, so no file upload is needed.

## Author

Karthik Somaraju
