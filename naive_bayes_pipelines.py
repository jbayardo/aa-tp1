from sklearn import *
import pandas
import nltk
import re
import email
from transforms import *


def extract_email_payloads(email):
    import html2text
    output = ''

    for part in email.walk():
        if part.get_content_type().startswith('text'):
            if part.get_content_type().endswith('html'):
                output += str(html2text.html2text(part.get_payload()))
            else:
                output += str(part.get_payload())
            output += '\n'

    return output


class EmailTokenizer():
    def __init__(self):
        # Tokenize into sentences using Punkt tokenizer
        # See: http://www.nltk.org/api/nltk.tokenize.html
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # Words are separated by whitespaces
        self.word_tokenizer = nltk.tokenize.RegexpTokenizer('\s+|\.+')

    def tokenize_email(self, email):
        output = []
        for sentence in self.sentence_tokenizer.tokenize(email):
            output.extend(self.word_tokenizer.tokenize(sentence))
        return output


if __name__ == '__main__':
    pipeline = sklearn.pipeline.Pipeline([
        ('transform_email', FunctionMapper(email.message_from_string)),
        ('extract_payload', FunctionMapper(extract_email_payloads)),
        # Using a TFIDF vectorizer will let us inversely weight common sequences
        ('generate_bag_of_words', sklearn.feature_extraction.text.TfidfVectorizer(
            ngram_range=(1, 16),
            use_idf=True,
            sublinear_tf=True,
            stop_words=nltk.corpus.stopwords.words('english'),
            analyzer='word',
            # This lets us tokenize emails however we see fit
            tokenizer=EmailTokenizer().tokenize_email
        )),
        # Helps prevent synonyms
        #('bow_pca', sklearn.decomposition.TruncatedSVD(n_components=2500)),
        #('select_best', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2,k=1000)),
        # Return the probability of spam
        ('train_naive_bayes', sklearn.naive_bayes.MultinomialNB())
    ])

    import load
    features, labels = load.load_dataset(sample=0.2)
    res = sklearn.cross_validation.cross_val_score(pipeline, features, labels, cv=5, scoring='precision', verbose=10)
    print(res)
