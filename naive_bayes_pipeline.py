from sklearn import *
from transforms import *
import nltk
import re
import email
import pandas

class Lemmatizer(sklearn.pipeline.BaseEstimator, sklearn.pipeline.TransformerMixin):
    def __init__(self):
        # This assumes we're using English in order to remove common words
        self.stopwords = nltk.corpus.stopwords.words('english')
        # Tokenize into sentences using Punkt tokenizer
        # See: http://www.nltk.org/api/nltk.tokenize.html
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # Words are separated by whitespaces
        self.word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        # Words are turned into their lemmas using WordNet
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def fit(self, x, y=None):
        return self

    # Partly inspired on https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
    def process(self, z):
        # Remove starting and ending whitespaces
        z = z.strip()

        # Merge repeated non word characters into single ones
        (z, replacements) = re.subn(r'(\W)\1+', r'\1', z)
        # Collapse more than three occurrences of the same letter into two (this assumes we're using English)
        (z, replacements) = re.subn(r'(\[a-zA-Z])\1{2,}', r'\1\1', z)
        # TODO: turn words with \/ into V, and so on
        output = []

        for sentence in self.sentence_tokenizer.tokenize(z):
            tokens = self.word_tokenizer.tokenize(sentence)
            tokens = nltk.pos_tag(tokens)

            for token, tag in tokens:
                # Avoid stopwords
                if token in self.stopwords:
                    continue

                # Translate into WordNet tag
                tag = {
                    'N': nltk.corpus.wordnet.NOUN,
                    'V': nltk.corpus.wordnet.VERB,
                    'R': nltk.corpus.wordnet.ADV,
                    'J': nltk.corpus.wordnet.ADJ
                }.get(tag[0], nltk.corpus.wordnet.NOUN)

                # Lemmatize
                lemma = self.lemmatizer.lemmatize(token, tag)
                output.append(lemma)

        return ' '.join(output)

    def transform(self, x, y=None):
        return [self.process(z) for z in x]

def extrP(s):
    try:
        return str(s.get_payload()[0])
    except:
        return ''

if __name__ == '__main__':
    pipeline = sklearn.pipeline.Pipeline([
        # TODO: header removal
        ('transform_email', FunctionMapper(email.message_from_string)),
        ('extract_payload', FunctionMapper(extrP)),
        # TODO: symbol removal
        # TODO: whitespace single character separation removal
        ('lemmatizer', Lemmatizer()),
        # Using a TFIDF vectorizer will let us inversely weight common sequences
        ('generate_bag_of_words', sklearn.feature_extraction.text.TfidfVectorizer(
            ngram_range=(1, 4),
            use_idf=True,
            sublinear_tf=True
        )),
        # Helps prevent synonyms
        #('bow_pca', sklearn.decomposition.TruncatedSVD(n_components=2500)),
        #('select_best', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2,k=1000)),
        # Return the probability of spam
        ('train_naive_bayes', sklearn.naive_bayes.MultinomialNB())
    ])

    # Load processed data
    dataset = pandas.read_msgpack('./data/development.msg', encoding='latin-1')
    import math

    dataset = dataset.sample(math.ceil(len(dataset) * 0.5))

    # Separate features and labels
    features = dataset['email'].values
    labels = dataset['class'].apply(lambda x: x == 1).values

    print(len(labels))
    res = sklearn.cross_validation.cross_val_score(pipeline, features, labels, cv=2, scoring='roc_auc', verbose=10)
    print(res)
