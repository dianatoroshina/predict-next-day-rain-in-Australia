import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing


data = pd.read_hdf('data-scaled.ready.h5', 'data')
print(data.head())

columns = list(data.drop(labels=['RainTomorrow', 'Date'], axis=1).columns)

scaler = preprocessing.StandardScaler()
data_transformed = pd.DataFrame(scaler.fit_transform(pd.DataFrame(data, columns=columns)))
columns_tranformed = data_transformed.columns
print(data_transformed.head())

data_transformed['Date'] = data['Date']
data_transformed['RainTomorrow'] = data['RainTomorrow']


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
reduce_mem_usage(data_transformed)


data_train, data_test = train_test_split(data_transformed, test_size=0.2)
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)
print(data_train.head())

x = pd.DataFrame(data_train, columns=columns_tranformed)

model = DecisionTreeClassifier(max_depth=16, max_features=26, min_samples_leaf=19, random_state=17)
model.fit(x, data_train['RainTomorrow'])

x_test = pd.DataFrame(data_test, columns=columns_tranformed)
data_test['target'] = model.predict(x_test)

print('Решающее дерево:', round(f1_score(data_test['RainTomorrow'], data_test['target']), 3))


bagging_model = BaggingClassifier(base_estimator=model, n_estimators=1000)
bagging_model.fit(x, data_train['RainTomorrow'])

data_test['target_bag'] = bagging_model.predict(x_test)
print('Бэггинг', round(f1_score(data_test['RainTomorrow'], data_test['target_bag']), 3))
