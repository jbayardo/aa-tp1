import email
from nltk.corpus import stopwords
import features
import pandas
from transforms import *
from sklearn import *

if __name__ == '__main__':
    pipeline = sklearn.pipeline.Pipeline([
        ('transform_email', FunctionMapper(email.message_from_string)),
        ('train_models', sklearn.pipeline.FeatureUnion([
            ('bag_of_words_model', sklearn.pipeline.Pipeline([
                # TODO: header removal
                ('extract_payload', FunctionMapper(str)),
                # TODO: symbol removal
                # TODO: whitespace single character separation removal
                # TODO: lemmatize
                # Using a TFIDF vectorizer will let us inversely weight common sequences
                ('generate_bag_of_words', sklearn.feature_extraction.text.TfidfVectorizer(
                    # Remove articles using NLTK's stopwords
                    stop_words=stopwords.words('english'),
                    ngram_range=(1, 5),
                    use_idf=True,
                    sublinear_tf=True)),
                # Helps prevent synonyms
                ('bow_pca', sklearn.decomposition.TruncatedSVD(n_components=1000)),
                ('select_best', sklearn.feature_selection.SelectKBest(
                    sklearn.feature_selection.chi2,
                    k=1000)),
                # Return the probability of spam
                ('train_naive_bayes', MultiProbNB())
            ])),
            ('other_models', sklearn.pipeline.Pipeline([
                ('generate_features', sklearn.pipeline.FeatureUnion([
                    ('content_type_features', FunctionTransformer(features.generate_content_type)),
                    ('email_counts_features', FunctionTransformer(features.generate_email_counts)),
                    ('case_ratio_features', FunctionTransformer(features.generate_upper_to_lower_case_ratios)),
                    ('email_chain_features', FunctionTransformer(features.generate_subject_is_chain)),
                    ('link_features', FunctionTransformer(features.generate_number_of_links)),
                    ('mailing_list_features', FunctionTransformer(features.generate_is_mailing_list))
                ])),
                ('replace_nans', sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean')),
                #('pca', sklearn.decomposition.RandomizedPCA(whiten=True)),
                ('train_other_models', sklearn.pipeline.FeatureUnion([
                    ('train_knn', ProbKNN()),
                    ('train_svm', ProbSVC(probability=True))
                ]))
            ]))
        ], n_jobs=1)),
        # We have 8 possibilities by this stage
        ('select_output_class', sklearn.ensemble.RandomForestClassifier(n_estimators=32, n_jobs=1))
    ])

    # Load processed data
    dataset = pandas.read_msgpack('./data/development.msg', encoding='latin-1')
    import math
    dataset = dataset.sample(math.ceil(len(dataset)*0.1))

    # Separate features and labels
    features = dataset['email'].values
    labels = dataset['class'].apply(lambda x: x == 1).values

    print(len(labels))
    res = sklearn.cross_validation.cross_val_score(pipeline, features, labels, cv=2, scoring='roc_auc', verbose=10, n_jobs=1)
    print(res)
