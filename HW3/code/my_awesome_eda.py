import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df, factor_threshold = 10):
    """
    Perform exploratory data analysis (EDA) on a given DataFrame.

    Parameters:
    - df (DataFrame): The input pandas DataFrame for analysis.
    - factor_threshold (int): Threshold to determine categorical variables. Default is 10.
    - id_column_name (str): Name of the column containing unique identifiers. Default is "PassengerId".

    Prints out various statistics and information related to the DataFrame:
    - Number of rows and columns in the DataFrame.
    - Summary statistics for numeric variables (sum, min, max, standard deviation, quartiles).
    - Value counts and percentages for categorical variables below the factor_threshold.
    - Total count of missing values (NaN).
    - Count of rows with missing values.
    - Number of missing values in each column.
    - Count of duplicate rows in the DataFrame.
    """
    print("Hello!\nHere is the results of EDA:\n")
    n_row = len(df)
    n_col = df.shape[1]
    print(f'Rows: {n_row}')
    print(f'Columns: {n_col}\n')
    
    numeric_columns = df.select_dtypes(include=np.number)
    for col in numeric_columns:
        if len(df[col].unique()) <= factor_threshold:
            numeric_columns = numeric_columns.drop(col, axis=1)
    n_sum = numeric_columns.sum(axis=0).round(2)
    n_min = numeric_columns.min(axis=0).round(2)
    n_max = numeric_columns.max(axis=0).round(2)
    n_std = numeric_columns.std(axis=0).round(2)
    q25 = numeric_columns.quantile(q=0.25).round(2)
    n_median = numeric_columns.quantile(q=0.5).round(2)
    q75 = numeric_columns.quantile(q=0.75).round(2)
    stats = pd.concat([n_sum, n_min, n_max, n_std, n_median, q25, q75], axis=1, keys=['sum', 'min', 'max', 'std', 'median', '0.25Q',  '0.75Q'])
    print(f'Numeric variables\n{stats}\n')
    
    categorical_columns = df.drop(numeric_columns, axis=1)
    for col in categorical_columns:
        if len(df[col].unique()) > factor_threshold:
            categorical_columns = categorical_columns.drop(col, axis=1)
        else:
            print(pd.concat([df[col].value_counts(), df[col].value_counts(normalize =True).round(2)], axis = 1), end = '\n\n')
    
    na_total = df.isna().sum().sum()
    na_rows = df.isna().sum(axis = 1)>0
    na_by_cols = df.isna().sum().to_string()
    print(f'Total NA: {na_total}')
    print(f'Rows with NA: {na_rows.sum()}')
    print(f'Number of NA\'s by columns:\n{na_by_cols}\n')
    duplicates = df.duplicated().sum()
    print(f'Duplicates: {duplicates}\n')