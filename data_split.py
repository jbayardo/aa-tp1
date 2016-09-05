import json
import pandas as pd
import numpy as np

# Leo los mails (poner los paths correctos).
ham_txt = json.load(open('./data/ham_dev.json'))
spam_txt = json.load(open('./data/spam_dev.json'))

txt = ham_txt + spam_txt
output = []

for (nro, text) in enumerate(txt):
    current = {}

    if nro <= len(ham_txt):
        # HAM es -1
        current['class'] = -1
    else:
        # SPAM es 1
        current['class'] = 1

    current['email'] = text
    output.append(current)

# This is the merged, created dataset
df = pd.DataFrame(output)

# Split and save holdout and training data
# We hold out about 10% of the data
mask = np.random.rand(len(df)) < 0.8

development = df[mask]
development.to_msgpack('./data/development.msg')

holdout = df[~mask]
holdout.to_msgpack('./data/holdout.msg')