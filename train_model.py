import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix

# Load processed data
dataset = pandas.read_msgpack('./data/processed.msg', encoding='latin-1')

# Separate features and labels
features = dataset[[x for x in dataset.columns if x != 'class']].values
labels = dataset['class'].apply(lambda x: x == 1)

# Train model
model = DecisionTreeClassifier()
res = cross_val_score(model, features, labels, cv=10, scoring='roc_auc')
print(np.mean(res), np.std(res))
