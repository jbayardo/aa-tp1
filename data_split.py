import json
import pandas as pd
import numpy as np
import email as em

# Leo los mails (poner los paths correctos).
ham_txt = json.load(open('./data/ham_dev.json'))
spam_txt = json.load(open('./data/spam_dev.json'))

txt = ham_txt + spam_txt
output = []

# Headers to include in attributes
whitelist = ['from', 'date', 'content-type', 'subject', 'mime-version', 'to', 'content-transfer-encoding', 'message-id', 'x-from', 'x-filename', 'x-origin', 'x-to', 'x-cc', 'x-bcc', 'x-priority', 'x-msmail-priority', 'x-mimeole', 'cc', 'bcc', 'received', 'x-mailer', 'reply-to', 'user-agent']

for (nro, text) in enumerate(txt):
    current = {}

    if nro <= len(ham_txt):
        current['class'] = 'ham'
    else:
        current['class'] = 'spam'

    email = em.message_from_string(text)

    # Include headers
    for (key, data) in email.items():
        if key in whitelist:
            current[key] = data
    
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
