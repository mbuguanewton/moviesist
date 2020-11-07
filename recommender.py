"""
    -> create a class recommender where all methods are located
    -> Since our recommender works without a dataset, we don't require pandas
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import  linear_kernel

"""
    arguments / parameters
        -> list of dictionaries
        -> useful fields
        -> main value to check

"""

class Recommender:
    def __init__(self):
        ...

    def get_cosine_matrix(self):
        ...

    def get_recommendations(self, title):
        ...





