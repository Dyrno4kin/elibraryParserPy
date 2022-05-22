import pandas as pd
import warnings
import numpy as np
from tqdm.auto import tqdm, trange
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, DBSCAN, MiniBatchKMeans
from flask import Flask, jsonify

app = Flask(__name__)

warnings.filterwarnings('ignore')

#Читаем из файла наш датасет
data = pd.read_csv('resourse/lenta-ru-news.csv', dtype={"url": "string", "title": "string", "text": "string", "topic": "string", "tags": "string", "date": "string"})
data.sample(5)
print(str(len(data)) + ' запросов считано')

# Задаем наши тематики
topics = ['Экономика', 'Мир', 'Наука и техника', 'Путешествия', 'Ценности']
target = ['Путешествия', 'Ценности', 'Экономика', 'Наука и техника', 'Мир']
news_in_cat_count = 400

df_res = pd.DataFrame()
for topic in tqdm(topics):
    df_topic = data[data['topic'] == topic][:news_in_cat_count]
    df_res = df_res.append(df_topic, ignore_index=True)
texts = df_res['text']

# Проверочная выборка на основе тематики
testTarget = []
for t in df_res['topic']:
    testTarget.append(target.index(t))


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")

def token_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

stopwords = nltk.corpus.stopwords.words('russian')
stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на'])

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

n_featur=200000
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000,
                                 min_df=0.01, stop_words=stopwords,
                                 use_idf=True, tokenizer=token_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

data1 = []
#Kmeans
km = KMeans(init='k-means++', n_clusters=5, random_state=1)
km.fit(tfidf_matrix)
data1.append(({
        'Algo': 'K-means',
        'ARI': metrics.adjusted_rand_score(testTarget, km.labels_),
        'AMI': metrics.adjusted_mutual_info_score(testTarget, km.labels_),
        'Homogenity': metrics.homogeneity_score(testTarget, km.labels_),
        'Completeness': metrics.completeness_score(testTarget, km.labels_),
        'V-measure': metrics.v_measure_score(testTarget, km.labels_)}))

#DBSCAN
dbscan = DBSCAN(eps = 1, min_samples = 5)
dbscan.fit(tfidf_matrix)
print(dbscan.labels_)
print(len(dbscan.labels_))
max(dbscan.labels_)
data1.append(({
        'Algo': 'DBSCAN',
        'ARI': metrics.adjusted_rand_score(testTarget, dbscan.labels_),
        'AMI': metrics.adjusted_mutual_info_score(testTarget, dbscan.labels_),
        'Homogenity': metrics.homogeneity_score(testTarget, dbscan.labels_),
        'Completeness': metrics.completeness_score(testTarget, dbscan.labels_),
        'V-measure': metrics.v_measure_score(testTarget, dbscan.labels_)}))

#SpectralClustering
spectral = SpectralClustering(n_clusters=5, random_state=1,affinity='rbf')
spectral.fit(tfidf_matrix)
print(spectral.labels_)
print(len(spectral.labels_))
max(spectral.labels_)
data1.append(({
        'Algo': 'SpectralClustering',
        'ARI': metrics.adjusted_rand_score(testTarget, spectral.labels_),
        'AMI': metrics.adjusted_mutual_info_score(testTarget, spectral.labels_),
        'Homogenity': metrics.homogeneity_score(testTarget, spectral.labels_),
        'Completeness': metrics.completeness_score(testTarget, spectral.labels_),
        'V-measure': metrics.v_measure_score(testTarget, spectral.labels_)}))

#Agglomerative
agglo = AgglomerativeClustering(n_clusters=5)
agglo_tfidf_matrix = tfidf_matrix.toarray()
agglo.fit(agglo_tfidf_matrix)
print(agglo.labels_)
print(len(agglo.labels_))
max(agglo.labels_)
data1.append(({
        'Algo': 'Agglomerative',
        'ARI': metrics.adjusted_rand_score(testTarget, agglo.labels_),
        'AMI': metrics.adjusted_mutual_info_score(testTarget, agglo.labels_),
        'Homogenity': metrics.homogeneity_score(testTarget, agglo.labels_),
        'Completeness': metrics.completeness_score(testTarget, agglo.labels_),
        'V-measure': metrics.v_measure_score(testTarget, agglo.labels_)}))

# MiniBatchKMeans
mbk  = MiniBatchKMeans(init ='k-means++', n_clusters = 5)
mbk.fit_transform(tfidf_matrix)
print(mbk.labels_)
print(len(mbk.labels_))
print(max(mbk.labels_))

data1.append(({
        'Algo': 'MiniBatchKMeans',
        'ARI': metrics.adjusted_rand_score(testTarget, mbk.labels_),
        'AMI': metrics.adjusted_mutual_info_score(testTarget, mbk.labels_),
        'Homogenity': metrics.homogeneity_score(testTarget, mbk.labels_),
        'Completeness': metrics.completeness_score(testTarget, mbk.labels_),
        'V-measure': metrics.v_measure_score(testTarget, mbk.labels_)}))

@app.route('/api/statistics', methods=['GET'])
def hello():
    return jsonify({'data': data1})

if __name__ == "__main__":
    app.run()