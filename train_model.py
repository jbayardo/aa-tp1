from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from transform_data_by_features import ds

dataset = ds[[x for x in ds.columns if x != 'class']].values
labels = ds['class'].apply(lambda x: x == 'spam')

model = DecisionTreeClassifier()
res = cross_val_score(model, dataset, labels, cv=10, scoring='roc_auc')
print(np.mean(res), np.std(res))
