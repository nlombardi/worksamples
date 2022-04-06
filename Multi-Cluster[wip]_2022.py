import numpy as np
import random
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
import warnings
import distance
import pandas as pd
import os
import time
import logging
from datetime import datetime

warnings.filterwarnings('error')
logging.basicConfig(filename='errors.log', level=logging.DEBUG)
path = os.environ['ONEDRIVE'] + "\\Documents\\2021\\Projects\\NLP_NN\\"
out_path = path + 'Aff_Prop_Output\\'
file = "Test_List.csv"


class Classify:

    def __init__(self, path=path, file=file, alg=0):
        self.words = pd.read_csv(path+file)
        self.alg = alg  # 0-aff_prop / 1-km

    def transform_list(self, word_list=False):
        # TODO: underlier_id_type to pre-process
        words_transformed = []
        if word_list:
            if type(word_list, list):
                self.words = pd.Series(word_list)
        self.words = self.words.drop_duplicates()
        for w1 in self.words.values:
            for w2 in w1:
                s1 = w2.split(',')
                for w3 in s1:
                    s2 = w3.split(';')
                    for w4 in s2:
                        words_transformed.append(w4)
        words_transformed = pd.Series(words_transformed).drop_duplicates().tolist()
        return words_transformed

    def get_word_list(self, word_list):
        if word_list:
            words2 = self.transform_list(word_list=word_list)
        else:
            words2 = self.transform_list()
        return words2

    def calc_levenshtein_dist(self, words, sample_size):
        words_array = np.asarray(random.sample(words, k=sample_size))
        print("Starting distance calculation..")
        lev_similarity = -1*np.array([[distance.levenshtein(w1, w2) for w1 in words_array] for w2 in words_array])
        print("Finished distance calculation..")
        return lev_similarity, words_array

    def run_km(self, lev_similarity, k_size):
        k = k_size
        km = KMeans(n_clusters=k, max_iter=100)
        km.fit(lev_similarity)
        return km if km else None

    def run_prop(self, lev_similarity):
        affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=None)
        affprop.fit(lev_similarity)
        return affprop if affprop else None

    def main(self, run_times=100, sample_size=2500, word_list=False):
        words2 = self.get_word_list(word_list=word_list)
        word_dict = {'Word': [], 'Matches': []}
        for i in range(run_times):
            start = time.time()
            lev_similarity, words_array = self.calc_levenshtein_dist(words=words2, sample_size=sample_size)
            try:
                if self.alg == 0:
                    cluster = self.run_prop(lev_similarity)
                    for cluster_id in np.unique(cluster.labels_):
                        word_dict['Word'].append(words_array[cluster.cluster_centers_indices_[cluster_id]])
                        word_dict['Matches'].append(np.unique(words_array[np.nonzero(cluster.labels_ == cluster_id)]))
                elif self.alg == 1:
                    # Note: for KM the clusters are not labelled as with Affinity Prop
                    cluster = self.run_km(lev_similarity, sample_size)
                    order_centroids = cluster.cluster_centers_.argsort()[:, ::-1]
                    for cluster_id in np.unique(cluster.labels_):
                        # print("Cluster ID: %d" % i)
                        # for j in order_centroids[i, :10]:
                        # TODO: need to fix the word append with results=km.fit_predict
                        word_dict['Word'].append(cluster_id)
                        word_dict['Matches'].append(np.unique([words_array[match]
                                                              for match in order_centroids[cluster_id, :60]]))
                else:
                    print("Wrong algorithm selected.")
                    raise Warning(f"Wrong input for algorithm type: alg={self.alg} is not a selection.")

            except Warning as e:
                logging.log(level=logging.DEBUG,
                            msg=f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")} Query failed to run.')
                logging.error(e)
                print("Convergence failed, see the log.")
            finally:
                end = time.time()
                runtime = (end - start) / 60
                logging.log(level=logging.INFO,
                            msg=f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")} Runtime: {runtime} minutes')
                print(f"Completed in: {(end - start) / 60} mins")
        return word_dict, cluster


class CompileApply:
    #TODO: connect the first class to this one, also allow for separate files as designed here

    def __init__(self, file_path: str = out_path):
        self.path = file_path
        self.files = [x for x in os.listdir(self.path) if x.find("file") != -1]

    def get_df(self, files):
        df = pd.DataFrame()
        for file in files:
            df_temp = pd.read_csv(self.path+file)
            if not df_temp.empty:
                df = df.append(df_temp)
        df = df.reset_index()
        df = df.drop(labels='Unnamed: 0', axis=1)
        return df

    def get_unique(self):
        df = self.get_df(self.files)
        multi_matches = df.groupby(['Word'])['Matches'].nunique().sort_values()
        words = [x for x in multi_matches.index]
        return words, df

    def merge_matches(self):
        words, df = self.get_unique()
        word_dict = {word: [] for word in words}
        for word in words:
            match_list = []
            results = df['Matches'].loc[df['Word'] == word]
            # clean and get the results
            for row in results:
                row_matches = row.replace("[", "").replace("]", "").replace("'", "").replace("\n", "")
                row_matches = row_matches.split(" ")
                for match in row_matches:
                    match_list.append(match)
            # drop the duplicates
            match_list = pd.Series(match_list).drop_duplicates().tolist()
            # loop through results and append to dict if the result is not in the list already
            for match in match_list:
                if match not in word_dict[word]:
                    word_dict[word].append(match)
        return word_dict

    def export_list(self):
        word_dict = self.merge_matches()
        df = pd.DataFrame.from_dict(word_dict, orient='index')
        df = df.transpose()
        df.to_excel(self.path+f'{self.files[0].strip(".csv")}.xlsx')


if __name__ == '__main__':
    # cluster = Classify(alg=0, file='DDR_IR_UNDER.csv')
    # output = cluster.main(sample_size=1000)
    # df = pd.DataFrame.from_dict(output)
    # df.to_csv(self.path+'IR_2022-01-06.csv')
    transform = CompileApply()
    transform.files = [x for x in os.listdir(out_path) if x.find("IR") != -1]
    transform.export_list()





