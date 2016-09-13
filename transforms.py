import sklearn


# Transforms any function into a map over the sample
class FunctionMapper(sklearn.pipeline.BaseEstimator, sklearn.pipeline.TransformerMixin):
    def __init__(self, function):
        self.function = function

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        print("Function: ", self.function.__name__)
        return [self.function(z) for z in x]


# Transforms any function into a map over the sample
class FunctionTransformer(sklearn.pipeline.BaseEstimator, sklearn.pipeline.TransformerMixin):
    def __init__(self, function):
        self.function = function

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        print("Function: ", self.function.__name__)
        model = sklearn.feature_extraction.DictVectorizer()
        return model.fit_transform([self.function(z) for z in x], y)


# Returns multinomial naive bayes probability for two classes
class MultiProbNB(sklearn.naive_bayes.MultinomialNB):
    def transform(self, x, y=None):
        return self.predict_proba(x)


class ProbSVC(sklearn.svm.SVC):
    def transform(self, x, y=None):
        return self.predict_proba(x)


class ProbKNN(sklearn.neighbors.KNeighborsClassifier):
    def transform(self, x, y=None):
        return self.predict_proba(x)