import pandas as pd
import numpy as np
import pickle
import math
import random
from scipy import stats


# import math
# import random
import itertools
import multiprocessing
from tqdm import tqdm
from time import time
import logging
import pickle

FOLDER_PATH = "dataset/yes_complete/"
MODEL_PATH = "model/"

# function to read txt data
def readTXT(filename, start_line=0, sep=None):
    with open(FOLDER_PATH+filename) as file:
        return [line.rstrip().split(sep) for line in file.readlines()[start_line:]]

# function to pickling model
def save2Pickle(obj, filename):
    with open(f"{MODEL_PATH}{filename}.pkl", "wb") as file:
        pickle.dump(obj, file)
        
# function to unpickling model
def loadPickle(filename):
    with open(f"{MODEL_PATH}{filename}.pkl", "rb") as file:
        return pickle.load(file)

# function to find k 
def locateOptimalElbow(x, y):
    # START AND FINAL POINTS
    p1 = (x[0], y[0])
    p2 = (x[-1], y[-1])
    
    # EQUATION OF LINE: y = mx + c
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = (p2[1] - (m * p2[0]))
    
    # DISTANCE FROM EACH POINTS TO LINE mx - y + c = 0
    a, b = m, -1
    dist = np.array([abs(a*x0+b*y0+c)/math.sqrt(a**2+b**2) for x0, y0 in zip(x,y)])
    return x[np.argmax(dist)]

# start recommending
# function to find mean vector
# def meanVectors(playlist,model):
#     vec = []
#     for song_id in playlist:
#         try:
#             vec.append(model.wv[song_id])
#         except KeyError:
#             continue
#     return np.mean(vec, axis=0)

# def similarSongsByVector(vec, n = 10, by_name = True, model, data):
#     # extract most similar songs for the input vector
#     similar_songs = model.wv.similar_by_vector(vec, topn = n)
    
#     # extract name and similarity score of the similar products
#     if by_name:
#         similar_songs = [(data.loc[song_id, "artist - title"], sim)
#                               for song_id, sim in similar_songs]
    
#     return similar_songs


# Evaluation
def hitRateRandom(playlist, n_songs, data):
    hit = 0
    for i, target in enumerate(playlist):
        random.seed(i)
        recommended_songs = random.sample(list(data.index), n_songs)
        hit += int(target in recommended_songs)
    return hit/len(playlist)

def hitRateContextSongTag(playlist, window, n_songs, data, mapping):
    hit = 0
    context_target_list = [([playlist[w] for w in range(idx-window, idx+window+1)
                             if not(w < 0 or w == idx or w >= len(playlist))], target)
                           for idx, target in enumerate(playlist)]
    for i, (context, target) in enumerate(context_target_list):
        context_song_tags = set(data.loc[context, 'tag_names'].explode().values)
        possible_songs_id = set(mapping[context_song_tags].explode().values)
        
        random.seed(i)
        recommended_songs = random.sample(possible_songs_id, n_songs)
        hit += int(target in recommended_songs)
    return hit/len(playlist)

def hitRateClustering(playlist, window, n_songs,objectmod, model, cluster):
    hit = 0
    context_target_list = [([playlist[w] for w in range(idx-window, idx+window+1)
                             if not(w < 0 or w == idx or w >= len(playlist))], target)
                           for idx, target in enumerate(playlist)]
    for context, target in context_target_list:
        cluster_numbers = objectmod.predict([model.wv[c] for c in context if c in model.wv.vocab.keys()])
        majority_voting = stats.mode(cluster_numbers).mode[0]
        possible_songs_id = list(cluster[cluster['cluster'] == majority_voting].index)
        recommended_songs = random.sample(possible_songs_id, n_songs)
        songs_id = list(zip(*recommended_songs))[0]
        hit += int(target in songs_id)
    return hit/len(playlist)

# def hitRateSong2Vec(playlist, window, n_songs):
#     hit = 0
#     context_target_list = [([playlist[w] for w in range(idx-window, idx+window+1)
#                              if not(w < 0 or w == idx or w >= len(playlist))], target)
#                            for idx, target in enumerate(playlist)]
#     for context, target in context_target_list:
#         context_vector = meanVectors(context)
#         recommended_songs = similarSongsByVector(context_vector, n = n_songs, by_name = False)
#         songs_id = list(zip(*recommended_songs))[0]
#         hit += int(target in songs_id)
#     return hit/len(playlist)