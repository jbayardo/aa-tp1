import email
import sklearn.pipeline
import sklearn.feature_extraction.text
import sklearn.tree
import sklearn.decomposition
import features
import pandas
import numpy as np

def mappable(f):
    def g(x):
        for entry in x:
            entry = f(email.message_from_string(entry))

        return x
    return g

pipeline = sklearn.pipeline.Pipeline([
    # TODO: avoid transforming into object for every function. Too expensive. should be able to do it just once
    #('transform_email', sklearn.preprocessing.FunctionTransformer(mappable(email.message_from_string), validate=False)),
    ('generate_features', sklearn.pipeline.FeatureUnion([
        ('content_type_features',
         sklearn.preprocessing.FunctionTransformer(mappable(features.generate_content_type), validate=False)),
        ('email_counts_features',
         sklearn.preprocessing.FunctionTransformer(mappable(features.generate_email_counts), validate=False)),
        ('case_ratio_features',
         sklearn.preprocessing.FunctionTransformer(mappable(features.generate_upper_to_lower_case_ratios),
                                                   validate=False)),
        ('email_chain_features',
         sklearn.preprocessing.FunctionTransformer(mappable(features.generate_subject_is_chain), validate=False)),
        ('link_features',
         sklearn.preprocessing.FunctionTransformer(mappable(features.generate_number_of_links), validate=False)),
        ('mailing_list_features',
         sklearn.preprocessing.FunctionTransformer(mappable(features.generate_is_mailing_list), validate=False)),
        ('bag_of_words_features', sklearn.pipeline.Pipeline([
            ('generate_bow', sklearn.feature_extraction.text.TfidfVectorizer()),
            ('pca', sklearn.decomposition.TruncatedSVD(n_components=200))
        ]))
    ])),
    ('train_tree', sklearn.tree.DecisionTreeClassifier())
])

# Load processed data
dataset = pandas.read_msgpack('./data/development.msg', encoding='latin-1')

# TODO: These three lines are just for fast iteration while testing
import numpy
mask = numpy.random.rand(len(dataset)) < 0.1
dataset = dataset[mask]

# Separate features and labels
features = dataset['email'].values
labels = dataset['class'].apply(lambda x: x == 1).values

res = sklearn.cross_validation.cross_val_score(pipeline, features, labels, cv=10, scoring='roc_auc')
print(res)