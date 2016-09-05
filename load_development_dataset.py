# Load up development dataset
import pandas as pd
import email

print("Loading data")
df = pd.read_msgpack('./data/development.msg', encoding='latin-1')
df['email'] = df['email'].apply(email.message_from_string)
