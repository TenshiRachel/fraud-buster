## Fraud Busters

This project aims to compare 6 classic learning algorithms to
predict fraudulent bank transactions.

### Algorithms used

1. Logistic Regression
2. Neural Networks
3. Decision Tree
4. Random Forest
5. Gradient Boosting (Adaptive Boosting)
6. Gradient Boosting (LightGBM)

### Datasets Used
[Bank Account Fraud Dataset Suite (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)

*Please download and keep ONLY Base.csv in a folder called 'data' 
in the project root directory*

Example:
![data example](https://github.com/TenshiRachel/fraud-buster/blob/master/img/data_example.png)

### Project Setup

1. Pull or Clone the project from GitHub
2. Open the terminal and enter the following command:
```commandline
pip install -r requirements.txt
```
3. Run main.py
4. (Optional) To view spread of data before and after SMOTE, go to src/process_data.py and uncomment the following:
```
X_resampled, y_resampled = get_resample(feature_engineering)
plot_fraud_distribution(X_train, y_train, X_resampled, y_resampled)
```

### Project Structure

```bash
|--data   # Contains the datasets
|
|--models # Contains the models used for the predictions
|
|--src    # Contains reusable codes
```
