import pandas as pd

base = pd.read_csv('./datasets/credit-data.csv') #load a file
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean() #replace all negative ages to ages mean

base.loc[pd.isnull(base['age'])] #find out on base all indexes with null age

forecasts = base.iloc[:, 1:4].values #forecasts variable will receive the columns 1,2 and 3 of base. The ":" means we want all lines.
classes = base.iloc[:, 4].values

#Using sklearn to localize all missing_values and replace using a strategy
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(forecasts[:,1:4])
forecasts[:,1:4] = imputer.transform(forecasts[:,1:4])

#when using knn algorithms is necessary standardisation or normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
forecasts = scaler.fit_transform(forecasts)

#Divide database in training data and test data
from sklearn.cross_validation import train_test_split
forecasts_training, forecasts_testing, classes_training, classes_testing = train_test_split(forecasts, classes, test_size= 0.25, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(forecasts_training, classes_training)

predicts = classifier.predict(forecasts_testing)

from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(classes_testing, predicts)

matrix = confusion_matrix(classes_testing, predicts)# Visualize how many records were correct per class
print(precision)

