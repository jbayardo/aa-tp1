def load_dataset(path = './data/development.msg', sample = 0.5, encoding = 'latin-1'):
    import pandas
    import math
    import numpy.random

    if sample <= 0 or sample > 1:
        raise Exception("Amount to sample must be between 0 and 1")

    # Load processed data
    print('[load_dataset] Loading dataset from', path)
    dataset = pandas.read_msgpack(path, encoding=encoding)
    print('[load_dataset] Shuffling dataset')
    dataset.reindex(numpy.random.permutation(dataset.index))
    print('[load_dataset] Sampling {0} ({1} samples) of the dataset'.format(sample, math.ceil(len(dataset) * sample)))
    dataset = dataset.sample(math.ceil(len(dataset) * sample))
    # Separate features and labels
    print ('[load_dataset] Separating dataset into features and labels')
    features = dataset['email'].values
    labels = dataset['class'].apply(lambda x: x == 1).values

    return features, labels