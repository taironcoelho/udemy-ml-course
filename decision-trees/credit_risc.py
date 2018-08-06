import pandas as pd

base = pd.read_csv('./datasets/credit-risc.csv')
forecasts = base.iloc[:, 0:4].values
classes = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
forecasts[:,0] = label_encoder.fit_transform(forecasts[:,0])
forecasts[:,1] = label_encoder.fit_transform(forecasts[:,1])
forecasts[:,2] = label_encoder.fit_transform(forecasts[:,2])
forecasts[:,3] = label_encoder.fit_transform(forecasts[:,3])


#Import Naive Bayes
from sklearn.tree import DecisionTreeClassifier, export

classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(forecasts, classes) #Generate Decision tree
print(classifier.feature_importances_)

export.export_graphviz(classifier,
                       out_file='tree.dot',
                       feature_names=['historia', 'divida', 'garantias', 'renda'],
                       class_names=['alto', 'moderado', 'baixo'],
                       filled=True,
                       leaves_parallel=True)

predict_result = classifier.predict([[0,0,1,2], [3,0,0,0]])

print(predict_result)
