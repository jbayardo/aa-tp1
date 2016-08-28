# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

# Leo los mails (poner los paths correctos).
ham_txt= json.load(open('../jsons_version/ham_txt.json'))
spam_txt= json.load(open('../jsons_version/spam_txt.json'))

# Imprimo un mail de ham y spam como muestra.
print ham_txt[0]
print "------------------------------------------------------"
print spam_txt[0]
print "------------------------------------------------------"

# Armo un dataset de Pandas 
# http://pandas.pydata.org/

df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

# Extraigo dos atributos simples: 
# 1) Longitud del mail.
df['len'] = map(len, df.text)

# 2) Cantidad de espacios en el mail.
def count_spaces(txt): return txt.count(" ")
df['count_spaces'] = map(count_spaces, df.text)

# Preparo data para clasificar
X = df[['len', 'count_spaces']].values
y = df['class']

# Elijo mi clasificador.
clf = DecisionTreeClassifier()

# Ejecuto el clasificador entrenando con un esquema de cross validation
# de 10 folds.
res = cross_val_score(clf, X, y, cv=10)
print np.mean(res), np.std(res)
# salida: 0.783040309346 0.0068052434174  (o similar)
