import email
import sklearn.pipeline
import sklearn.feature_extraction.text
import sklearn.tree
import sklearn.decomposition
import features
import pandas
import numpy as np


# Transforms any function into a map over the sample
class FunctionMapper(sklearn.pipeline.BaseEstimator, sklearn.pipeline.TransformerMixin):
    def __init__(self, function):
        self.function = function

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return map(self.function, x)


# Transforms any function into a map over the sample
class FunctionTransformer(sklearn.pipeline.BaseEstimator, sklearn.pipeline.TransformerMixin):
    def __init__(self, function):
        self.function = function
        self.model = sklearn.feature_extraction.DictVectorizer()

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return self.model.fit_transform(map(self.function, x), y)


if __name__ == '__main__':
    pipeline = sklearn.pipeline.Pipeline([
        ('transform_email', FunctionMapper(email.message_from_string)),
        ('generate_features', sklearn.pipeline.FeatureUnion([
            ('content_type_features', FunctionTransformer(features.generate_content_type)),
            ('email_counts_features', FunctionTransformer(features.generate_email_counts)),
            ('case_ratio_features', FunctionTransformer(features.generate_upper_to_lower_case_ratios)),
            ('email_chain_features', FunctionTransformer(features.generate_subject_is_chain)),
            ('link_features', FunctionTransformer(features.generate_number_of_links)),
            ('mailing_list_features', FunctionTransformer(features.generate_is_mailing_list)),
            ('bag_of_words_features', sklearn.pipeline.Pipeline([
                ('extract_payload', FunctionMapper(str)),
                ('generate_bow', sklearn.feature_extraction.text.TfidfVectorizer()),
                ('pca', sklearn.decomposition.TruncatedSVD(n_components=200))
            ]))
        ], n_jobs=2)),
        ('replace_nans', sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean')),
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
