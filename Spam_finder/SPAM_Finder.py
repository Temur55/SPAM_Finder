import pandas as pd
import time
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def write_out(name, out):
    with open(name+'.txt', 'w') as file:
        file.write(str(out))
    file.close()

def splt(data):
    lst1, lst2 = [], []
    for i in data:
        lst1.append(i[0])
        lst2.append(i[1])
    return lst1, lst2


def my_pipeline(n1=2, n2=2, cls=LogisticRegression, need_rs=True, vect=CountVectorizer):
    vectorizer = vect(ngram_range=(n1, n2))
    features = vectorizer.fit_transform(feedback)

    if need_rs:
        model = cls(random_state=2)
    else:
        model = cls()
    res = cross_val_score(model, features, label_num, scoring="f1", cv=10, n_jobs=-1)

    return (np.mean(res))


with open('SMSSpamCollection.txt', encoding='utf-8') as text:
    data = text.read().rstrip().split('\n')
    for i in range(len(data)):
        data[i] = data[i].split('\t')

label, feedback = splt(data)

label = np.array(label)
label_num = np.where(label == "spam", 1, 0)


vectorizer = CountVectorizer()
features = vectorizer.fit_transform(feedback)


# Проверим качество модели на 10 фолдах
cross_val_vect = cross_val_score(LogisticRegression(random_state=2), features, label_num, scoring="f1", cv=10, n_jobs=-1)

# Сохраним среднее качество
outOne = np.mean(cross_val_vect)

cls = LogisticRegression(random_state=2)
cls.fit(features, label_num)

# Проверим работу на тестовых данных
test = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! Subscribe6GB",

        "FreeMsg: Txt: claim your reward of 3 hours talk time",

        "Have you visited the last lecture on physics?",

        "Have you visited the last lecture on physics? Just buy this book and you will have all materials! Only 99$",

        "Only 99$"]

test_features = vectorizer.transform(test)

# Предсказания модели на тесте
prediction = cls.predict(test_features)


"""
Посмотрим на работу Vectorizer при различных значениях k и n в k-skip-n-grams
и разных моделей МО
"""

outTwo = []
for i in prediction:
    outTwo.append(str(i))

resultsForThree = []
for i in [[2,2],[3,3],[1,3]]:
    resultsForThree.append(my_pipeline(i[0],i[1]))

resultsForFour = []
for i in [[2,2],[3,3],[1,3]]:
    resultsForFour.append(my_pipeline(i[0],i[1], MultinomialNB, need_rs=False))

outThree = []
for i in resultsForThree:
    outThree.append(str(round(i,2)))

outFour = []
for i in resultsForFour:
    outFour.append(str(round(i,2)))

# Выведем результаты
write_out('1', round(np.mean(prediction), 1))
write_out('2', " ".join(outTwo))
write_out('3', " ".join(outThree))
write_out('4', " ".join(outFour))
