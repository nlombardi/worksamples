import logging
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import os
import re
import random
import pickle
import time
import warnings

# for summarizing
from docx import Document
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

"""
DATA Source(s)
__________________________________________
"""
path = os.environ['ONEDRIVE'] + "\\Documents\\2021\\Projects\\NLP_NN\\"
file = "Test_List.csv"
cdor_data = "CDOR_DATA\\cdor_data.csv"
# Get the word database for NLTK


"""
CDOR LABELS for Supervised NN
__________________________________________
"""
df = pd.read_csv(path + cdor_data, parse_dates=True)

cdor_ref = {'Index': [], 'Leg_1_Rate': [], 'Leg_2_Rate': []}
for i, l1, l2 in zip(df.index, df['Leg_1_Floating_Rate_Index'], df['Leg_2_Floating_Rate_Index']):
    found = False
    if re.search('CDOR', str(l1)):
        found = True
    elif re.search('CDOR', str(l2)):
        found = True

    if found:
        cdor_ref['Index'].append(i)
        cdor_ref['Leg_1_Rate'].append(l1)
        cdor_ref['Leg_2_Rate'].append(l2)

df_ref = pd.DataFrame.from_dict(cdor_ref)

"""
Text Pre-Processing
------------------------------------------
Remove punctuation, stopwords: ['and', 'in', etc.], semi-colons, and returns a list of words
"""

test_list = pd.read_csv(path+file)
clean_list = []

def split_text(text):
    delimiters = [',', ';']
    split_list = []
    for delimiter in delimiters:
        split = [x for x in text.split(delimiter) if re.search(delimiter, text)]
        if split:
            split_list.append(split[0])
    return split_list


# def clean_text(text):
#     nltk.download('wordnet')
#     lemmy = WordNetLemmatizer()
#     # cleaned = re.sub(r'(^\W|\W$)', '', text)  # To replace periods if needed


for asset_id in test_list['Underlying_Asset_ID']:
    basket_split = split_text(asset_id)
    if basket_split:
        for asset in basket_split:
            clean_list.append(asset)
    else:
        clean_list.append(asset_id)


"""

********* K-MEANS ALGORITHM WITH ELBOW METHOD FOR OPTIMAL K *************

"""


# Find elbow
def get_slopes(elbow_data):
    slopes = {'K': [], 'Slope': []}
    for ssd in range(len(elbow_data['SSD'])-2):
        # take difference of first two points to get slope
        slopes['K'].append(elbow_data['K'][ssd+1])
        slopes['Slope'].append((elbow_data['SSD'][ssd+1] - elbow_data['SSD'][ssd])/elbow_data['SSD'][ssd])
    return slopes


def find_elbow(slopes):
    # find where the max-k difference from the mean is < -0.065 (for test 1 was  1317)
    df1 = pd.DataFrame.from_dict(slopes)
    mean = df1['Slope'].mean()
    df1['diff_from_mean'] = [(lambda x: x-mean)(x) for x in df1['Slope']]
    """
    looks for the minimum K that has a difference from the mean slope that is < -0.065 (where the bend starts)
    needs to be in the upper quantiles because of outliers in first few SSD's
    """
    elbow = min(df1['K'].loc[(df1['diff_from_mean'] < -0.065) & (df1['K'] > df1['K'].quantile(0.5))])
    return elbow


def get_elbow_data():
    # define a sample from the clean list of 5,000
    small_test_list = [(lambda x: x)(random.choice(clean_list)) for x in range(0, 5000)]
    # # Split train and test datasets ***unnecessary for unsupervised models (kmeans)
    # train = small_test_list[:int(len(small_test_list)*2/3)]
    # test = small_test_list[len(train):]

    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    vect_list = vectorizer.fit_transform(small_test_list)
    vectorizer.get_feature_names()
    # print(vect_list.shape)

    # find optimal # of clusters using elbow method ***1317 distinct clusters
    sum_of_squared_distances = []
    K = range(1, vect_list.shape[1])

    print("Starting KMeans....")
    for k in K:
        print(f"\t k={k}")
        with warnings.catch_warnings(record=True) as w:
            km = KMeans(n_clusters=k, max_iter=10)
            km = km.fit(vect_list)
            sum_of_squared_distances.append(km.inertia_)
            if w[-1].category == ConvergenceWarning:
                break

    elbow_data = {'K': K, 'SSD': sum_of_squared_distances}
    return elbow_data


elbows = []

for i in range(int(round(len(clean_list)/5000, 0))):
    start = time.time()
    print(f"Starting: {i}")
    elbow_data = get_elbow_data()
    slopes = get_slopes(elbow_data)
    elbows.append(find_elbow(slopes))
    end = time.time()
    print(f"Took {(end-start)/(60*60)} hours to run {i}")
    with open(path+f'elbow_data_{i}.p', 'wb') as f:
        pickle.dump(elbow_data, f)
    f.close()

# with open(path+'eblows.p', 'wb') as f:
#     pickle.dump(elbows, f)
# f.close()


plt.close()

# # Show the plot to find the elbow (math following)
# plt.plot(K, sum_of_squared_distances, 'bx-')
# plt.xlabel('k')
# plt.ylabel('sum_of_squared_distances')
# plt.show()

"""
TEMP: load elbow data
"""
elbow_temp = {'K': [], 'SSD': []}
elbow_data_list = 0

with os.scandir(path) as f:
    for file in f:
        if file.name.endswith('p') and file.name.startswith('elbow'):
            elbow_data_list += 1


for i in range(elbow_data_list-1):
    with open(path+f'elbow_data_{i}.p', 'rb') as f:
        elbow_data = pickle.load(f)
    elbow_temp['K'].append(elbow_data['K'])
    elbow_temp['SSD'].append(elbow_data['SSD'])

elbows = [(lambda y: find_elbow(get_slopes(y)))(y={'K': x[0], 'SSD': x[1]}) for x in zip(elbow_temp['K'], elbow_temp['SSD'])]
s1 = pd.Series(elbows)
avg_elbow = int(round(s1.mean(), 0))

# GRAPH AND EXPORT GRAPH
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
# max([x for x in elbows[key][0]['SSD'] for key in elbows])
for i in range(8):
    with open(path + f'elbow_data_{i}.p', 'rb') as f:
        elbows[f'eb{i}'].append(pickle.load(f))
    f.close()

for key in elbows:
    color = colors[list(elbows.keys()).index(key)]
    plt.plot(elbows[key][0]['K'], elbows[key][0]['SSD'], color=color)
plt.axvline(x=1717, ymin=-10, ymax=3000, color='g', linestyle=':', linewidth=2)
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
# xticks = np.arange(0, 2000, 200)
# plt.xticks(xticks)
yticks = np.arange(0, 3000, 200)
plt.yticks(yticks)
plt.ylim(top=3000)
plt.tight_layout(pad=0.3)
plt.savefig(path+'Elbow.png')
plt.show()

results = km.fit_predict(lev_similarity) # for KM
results = ap.labels_ # for ap


for i in range(len(km.cluster_centers_)):
    plt.scatter(lev_similarity[results==i, 0], lev_similarity[results==i, 1], s=100, c=f'C{i}')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters of Customers')
plt.show()
"""

SUMMARIZE TEXT USING NLTK (or PySummarize)

"""
path = os.environ['ONEDRIVE'] + "\\Documents\\2021\\Ad-Hoc\\Wealthsimple\\Fractional_Shares_Analysis\\"
file = "Wealthsimple Trade - Fractional Share Offering Summary.docx"


# Load Word Doc
f = open(path+file, 'rb')
doc = Document(f)
f.close()
doc_text = ' \n'.join([x.text for x in doc.paragraphs])
# Object of automatic summarization.
auto_abstractor = AutoAbstractor()
# Set tokenizer.
auto_abstractor.tokenizable_doc = SimpleTokenizer()
# Set delimiter for making a list of sentence.
auto_abstractor.delimiter_list = [".", "\n"]
# Object of abstracting and filtering document.
abstractable_doc = TopNRankAbstractor()
# Summarize document.
result_dict = auto_abstractor.summarize(doc_text, abstractable_doc)