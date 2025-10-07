import pandas as pd
import numpy as np
from functools import reduce

def explore():

    print(f"pandas version is {pd.__version__}")
    
    input_path = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
    df = pd.read_csv(input_path)
    
    
    print(f'number of rows = {df.shape[0]}')
    
    print(f'number of fuel types = {df["fuel_type"].nunique()}')
    
    miss_val_cols = reduce(lambda acc, cur_col: acc +  (1 if df[cur_col].isna().sum() > 0 else 0), df.columns, 0)
    print(f'number of columns in the dataset with missing values is {miss_val_cols}')

    
    max_eff = df[df['origin'].str.upper() == 'ASIA']['fuel_efficiency_mpg'].max()
    print(f'max fuel efficiency for Asia is {max_eff}')

    median = df['horsepower'].median()
    print(f'median = {median}')
    
    most_freq_hp = df["horsepower"].mode()[0]
    print("Most frequent horsepower:", most_freq_hp)

    most_freq_series = df["horsepower"].fillna(most_freq_hp)
    updated_median = most_freq_series.median()
    print(f'updated_median = {updated_median}')
    print(f'median changed? - {"no" if median==updated_median else ("Yes, it increased" if updated_median > median else "Yes, it decreased")}')

    
    X = df[df["origin"].str.upper() == 'ASIA'][["vehicle_weight", "model_year"]].head(7).to_numpy()
    
    XT = X.T
    XTX = np.dot(XT, X)

    inv_XTX = np.linalg.pinv(XTX)

    Y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

    w = np.dot(np.dot(inv_XTX, XT), Y)
    
    sum_el = w.sum()
    print(sum_el)



def main():
    explore()

if __name__ == "__main__":
    main()