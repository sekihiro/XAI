#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from glob import glob
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import MeCab
import re
from gensim.models import Word2Vec
import numpy as np
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_PATH = '/home/seki/_seki/XAI'
TXT_PATH = BASE_PATH + '/text2'
MODEL_PATH = BASE_PATH + '/livedoor_corpus_feature200.model'
FIG_PATH = BASE_PATH + '/shap.png'

tagger = MeCab.Tagger("-Owakati")

# 形態素解析の定義
def make_wakati(sentence):
    sentence = tagger.parse(sentence)
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=＝)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    wakati = sentence.split(" ")
    wakati = list(filter(("").__ne__, wakati))
    return wakati

# 文書ベクトル作成用関数の定義
def wordvec2docvec(sentence):
    # 文章ベクトルの初期値（0ベクトルを初期値とする）
    docvecs = np.zeros(num_features, dtype="float32")

    # 文章に現れる単語のうち、モデルに存在しない単語をカウントする
    denomenator = len(sentence)

    # 文章内の各単語ベクトルを足し合わせる
    for word in sentence:
        try:
            temp = model[word]
        except:
            denomenator -= 1
            continue
        docvecs += temp

    # 文章に現れる単語のうち、モデルに存在した単語の数で割る
    if denomenator > 0:
        docvecs =  docvecs / denomenator

    return docvecs

# カテゴリ（ディレクトリ名）をリスト化
categories = [name for name in os.listdir(TXT_PATH) if os.path.isdir(TXT_PATH + "/" +name)]
print(categories)

# カテゴリをID化
category2id = {}
for i, cat in enumerate(categories):
    category2id[cat] = i

# DataFrame作成
datasets = pd.DataFrame(columns=["document", "category"])

for category in tqdm(categories):
    path = TXT_PATH + "/" + category + "/*.txt"
    files = glob(path)
    for text_name in files:
        with open(text_name, 'r', encoding='utf-8') as f:
            document = f.read()
            row = pd.Series([document, category], index=datasets.columns)
            datasets = datasets.append(row, ignore_index=True)
print("doc num", len(datasets))

# test
print(make_wakati(datasets["document"][0]))

# word2vec parameters
num_features = 200
min_word_count = 5
num_workers = 40
context = 10
downsampling = 1e-3

# コーパス読み込み
corpus = []
print("loading corpus ...")
for doc in tqdm(datasets["document"]):
    corpus.append(make_wakati(doc))

if os.path.isfile(MODEL_PATH):
    # word2vecモデルの読み込み
    print("loading word2vec model ...")
    model = Word2Vec.load(MODEL_PATH)
else:
    # word2vecモデルの作成＆モデルの保存
    print("cleating word2vec model ...")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(corpus, workers=num_workers, hs=0, sg=1, negative=10, iter=25, size=num_features, min_count = min_word_count,  window = context, sample = downsampling, seed=1)
    model.save(MODEL_PATH)
    print("Done.")

# test
print(model.wv.most_similar("男性"))

print(len(datasets["document"]))
X, Y = [], []
for doc, category in tqdm(zip(datasets["document"], datasets["category"])):
    wakati = make_wakati(doc)
    docvec = wordvec2docvec(wakati)
    X.append(list(docvec))
    Y.append(category2id[category])
data_X = pd.DataFrame(X, columns=["X" + str(i + 1) for i in range(num_features)])
data_Y = pd.DataFrame(Y, columns=["category_id"])

train_x, test_x, train_y, test_y = train_test_split(data_X, data_Y, train_size= 0.7)

# XGBoostで分類器を作成＆予測
print("Fitting XGboost model ...")
xgb_model = xgb.XGBClassifier()
xgb_model.fit(train_x, train_y)
print("Done.")

# 予測
pred = xgb_model.predict(test_x)
print(classification_report(pred, test_y["category_id"], target_names=categories))


# SHAP
import shap
fig = plt.figure()
shap.initjs()
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(train_x)
plt = shap.summary_plot(shap_values, train_x)
fig.savefig(FIG_PATH)

print(' ----- finish ----- ')

# EOF #
