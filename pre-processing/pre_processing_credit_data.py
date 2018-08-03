import pandas as pd
base = pd.read_csv('credit-data.csv') #load a file
base.describe() #print statistics about the base
base.loc[base['age'] < 0] #find out on base all indexes with negative age

#base.drop('age', 1, inplace=True) Command to drop a column in base.
#base.drop(base[base.age < 0].index, inplace=True) Command to drop records with negative age

base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean() #replace all negative ages to ages mean

base.loc[pd.isnull(base['age'])] #find out on base all indexes with null age

forecasts = base.iloc[:, 1:4].values #forecasts variable will receive the columns 1,2 and 3 of base. The ":" means we want all lines.
config_class = base.iloc[:, 4].values

print(config_class)

#Using sklearn to localize all missing_values and replace using a strategy
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(forecasts[:,0:3])
forecasts[:,0:3] = imputer.transform(forecasts[:,0:3])

#when using knn algorithms is necessary standardisation or normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
forecasts = scaler.fit_transform(forecasts)
