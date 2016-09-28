import email
import features
import pandas
import nltk
import re
from transforms import *
from sklearn import *

if __name__ == '__main__':
    pipeline = sklearn.pipeline.Pipeline([
        ('transform_email', FunctionMapper(email.message_from_string)),
        ('train_models', sklearn.pipeline.FeatureUnion([
            ('bag_of_words_model', sklearn.pipeline.Pipeline([
                ('extract_payload', FunctionMapper(features.extract_email_payloads)),
                # Using a TFIDF vectorizer will let us inversely weight common sequences
                ('generate_bag_of_words', sklearn.feature_extraction.text.TfidfVectorizer(
                    ngram_range=(1, 16),
                    use_idf=True,
                    sublinear_tf=True,
                    stop_words=nltk.corpus.stopwords.words('english'),
                    analyzer='word',
                    # This lets us tokenize emails however we see fit
                    tokenizer=features.EmailTokenizer().tokenize_email
                )),
                # Helps prevent synonyms
                #('bow_pca', sklearn.decomposition.TruncatedSVD(n_components=2500)),
                #('select_best', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2,k=1000)),
                # Return the probability of spam
                ('train_naive_bayes', sklearn.naive_bayes.MultinomialNB())
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
    import load
    features, labels = load.load_dataset(sample=0.2)

    print(len(labels))
    res = sklearn.cross_validation.cross_val_score(pipeline, features, labels, cv=2, scoring='roc_auc', verbose=10, n_jobs=1)
    print(res)
