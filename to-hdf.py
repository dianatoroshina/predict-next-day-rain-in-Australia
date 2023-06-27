import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


data = pd.read_csv("weatherAUS.csv")
print(data.head())


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == 'int':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("i1").min and c_max < np.iinfo("i1").max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo("i2").min and c_max < np.iinfo("i2").max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo("i4").min and c_max < np.iinfo("i4").max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo("i8").min and c_max < np.iinfo("i8").max:
                df[col] = df[col].astype(np.int64)
        elif col == 'timestamp':
            df[col] = pd.to_datetime(df[col])
        elif str(col_type)[:8] != 'datetime':
            df[col] = df[col].astype("category")
            
    end_mem = df.memory_usage().sum() / 1024**2
    print(start_mem, end_mem)
    print('Потребление памяти меньше на', round(start_mem - end_mem, 2), 
          "Мб (минус", round(100 * (start_mem - end_mem) / start_mem, 1), '%)')
    return df


data = reduce_mem_usage(data)
print(data.info())

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
for column in data.columns:
    if column != ('RainToday' or 'RainTomorrow'):
        data[column] = imputer.fit_transform(data[column].values.reshape(-1,1))[:,0]

most_common_raintoday = data['RainToday'].mode()[0]
data['RainToday'].fillna(most_common_raintoday, inplace=True)

columns = data.drop(labels='RainTomorrow', axis=1).columns
columns = list(columns)
print(columns)
columns_category = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
columns_numeric = []

for column in columns:
    if column not in columns_category:
        columns_numeric.append(column)
        
columns_numeric.remove('Date')   
print(columns_numeric)
print(columns_category)

data['RainTomorrow'] = data['RainTomorrow'].apply(lambda x: 1 if x=='Yes' else 0)
print(data.head())

for col in columns_category:
    for elem in data[col].unique():
        data[col + str(elem)] = data[col].isin([elem]).astype('int8')
print(data.head())

data.drop(labels=columns_category, axis=1, inplace=True)
columns = list(data.drop(labels=['RainTomorrow', 'Date'], axis=1).columns)

data.to_hdf('data-scaled.ready.h5', 'data', format='table', complevel=9, mode='w')


data = pd.read_hdf('data-scaled.ready.h5', 'data')
print(data.head())
