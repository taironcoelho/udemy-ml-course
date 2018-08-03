import pandas as pd

base = pd.read_csv('census.csv')

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

#create dummy variables to not get wrong interpretations about categorical variables
one_hot_encoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
forecasts = one_hot_encoder.fit_transform(forecasts).toarray()

label_encoder_class = LabelEncoder()
classes = label_encoder_class.fit_transform(classes)

#when using knn algorithms is necessary standardisation or normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
forecasts = scaler.fit_transform(forecasts)