import pandas as pd

def evaluation_data(model_type: str, folder: str, index: str):
    df = pd.read_csv(f"Optimizer/data_optimization/{model_type}/{folder}/{model_type}_{index}_hyperparameters_results.csv").fillna('None')

    groupby_cols = ['lag', 'dif_all', 'train_size', 'buckets', 'TI']
    results = {col: get_results(df, col) for col in groupby_cols}

    for col in groupby_cols:
        print(f"Results for {col}:")
        print(results[col])

def get_results(df, groupby_col):
    agg_results = df.groupby(groupby_col).agg({'bin_accuracy': ['mean', 'max', 'min', 'count']}).reset_index()
    agg_results.columns = [groupby_col, 'Mean', 'High', 'Low', 'Count']
    return agg_results