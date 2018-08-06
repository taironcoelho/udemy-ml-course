import pandas as pd

base = pd.read_csv('./datasets/census.csv')

forecasts = base.iloc[:,0:14].values

classes = base.iloc[:,14].values

#Encode string labels in num values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_forecasts = LabelEncoder()
forecasts[:,1] = label_encoder_forecasts.fit_transform(forecasts[:,1])
forecasts[:,3] = label_encoder_forecasts.fit_transform(forecasts[:,3])
forecasts[:,5] = label_encoder_forecasts.fit_transform(forecasts[:,5])
forecasts[:,6] = label_encoder_forecasts.fit_transform(forecasts[:,6])
forecasts[:,7] = label_encoder_forecasts.fit_transform(forecasts[:,7])
forecasts[:,8] = label_encoder_forecasts.fit_transform(forecasts[:,8])
forecasts[:,9] = label_encoder_forecasts.fit_transform(forecasts[:,9])
forecasts[:,13] = label_encoder_forecasts.fit_transform(forecasts[:,13])

# This combinated with standard scaler decrease the precision
# one_hot_encoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
# forecasts = one_hot_encoder.fit_transform(forecasts).toarray()

label_encoder_class = LabelEncoder()
classes = label_encoder_class.fit_transform(classes)

#when using knn algorithms is necessary standardisation or normalization
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# forecasts = scaler.fit_transform(forecasts)

#Divide database in training data and test data
from sklearn.cross_validation import train_test_split
forecasts_training, forecasts_testing, classes_training, classes_testing = train_test_split(forecasts, classes, test_size= 0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(forecasts_training, classes_training)

predicts = classifier.predict(forecasts_testing)

from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(classes_testing, predicts)

matrix = confusion_matrix(classes_testing, predicts)# Visualize how many records were correct per class
print(precision)