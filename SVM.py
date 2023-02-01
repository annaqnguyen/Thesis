import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import graphviz 
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


detections = pd.read_csv(r"D:\Anna\Documents\Uni\Thesis\Code\Detections\Q0000012_results.csv")
train_data = detections.drop(['Date', 'Time', 'Label', 'Trend', 'Error'], axis = 1)
# for i in range(len(train_data)):
#     row = detections.iloc[i]
#     train_data.at[i,'Magnitude']=row[3] / row[8]
#     train_data.at[i,'Before']=row[4] / row[8]
#     train_data.at[i,'After']=row[5] / row[8]
# sc = StandardScaler()
# train_data = sc.fit_transform(train_data)

train_label = detections['Label']

test = pd.read_csv(r"D:\Anna\Documents\Uni\Thesis\Code\Detections\Q0000014_results.csv")
test_data = test.drop(['Date', 'Time', 'Label', 'Trend', 'Error'], axis = 1)
# for i in range(len(test_data)):
#     row = detections.iloc[i]
#     test_data.at[i,'Magnitude']=row[3] / row[8]
#     test_data.at[i,'Before']=row[4] / row[8]
#     test_data.at[i,'After']=row[5] / row[8]

test_label = test['Label']

# Create training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# X_train = X.head(33)
# X_test = X.tail(len(detections) - 33)
# y_train = y[:33]
# y_test = y[33:]


#SVM

print("SVM")
svclassifier = SVC(kernel='linear')

# Fit the data, train the model
svclassifier.fit(train_data, train_label)


# Do the prediction from the trained model
y_pred = svclassifier.predict(test_data)
print(y_pred)


#KNN
# print(("KNN"))
# # test_data = sc.transform(test_data)
# KNN_model = KNeighborsClassifier(n_neighbors=5)
# KNN_model.fit(train_data, train_label)
# KNN_prediction = KNN_model.predict(test_data)
# print(KNN_prediction)

#Decision Trees

# print("Decision Trees")
# sc = StandardScaler()
# train_data_scaled = sc.fit_transform(train_data)
# test_data_scaled = sc.transform(test_data)

# classifier = DecisionTreeClassifier()
# classifier.fit(train_data, train_label)
# tree.plot_tree(classifier) 
# dot_data = tree.export_graphviz(classifier, out_file=None, feature_names=train_data.columns.tolist(),class_names=['0', '1', '2', '3', '4', '5'], filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data) 
# graph.render("class") 
# y_pred = classifier.predict(test_data)
# print(y_pred)



# Random Forest

# print("Random Forest")
# sc = StandardScaler()
# train_data_scaled = sc.fit_transform(train_data)
# test_data_scaled = sc.transform(test_data)

# classifier = RandomForestClassifier(n_estimators=20, random_state=0)
# classifier.fit(train_data_scaled, train_label)
# tree.plot_tree(classifier.estimators_[0]) 
# dot_data = tree.export_graphviz(classifier.estimators_[0], out_file=None, feature_names=train_data.columns.tolist(),class_names=['0', '1', '2', '3', '4', '5'], filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(dot_data) 
# graph.render("class") 

# y_pred = classifier.predict(test_data_scaled)
# print(y_pred)