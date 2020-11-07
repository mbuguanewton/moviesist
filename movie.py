# flask movie recommender system
#imports
from flask import Flask,render_template,request,url_for

##################################################################
import pandas as pd

movies1 = pd.read_csv("datasets/IMDB-Movie-Data.csv")
movies2 = pd.read_csv("datasets/tmdb-movies.csv")
movies3 = pd.read_csv("datasets/movies.csv")

Movies = [movies1,movies2,movies3]
result = pd.concat(Movies)
#movies.head()
useful = ["Title", "Genre", "Description", "Rating"]
##useful = ["original_title", "genres", "overview", "vote_average"]
df = result[useful]
#print(df.shape)
df= df.fillna('')

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
import time

t1 = time.time()

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
#metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df["Description"])

t2 = time.time()
#print(t2 - t1)
#Output the shape of tfidf_matrix
#tfidf_matrix.shape
#tfidf_matrix

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
# Function that takes in movie title as input and outputs most similar movies

def get_recommendations(title, cosine_sim=cosine_sim):

    B = title.split()
    D = ''
    l = len (B)
    s = l-1
    B1 = B[:s]
    for i in B1:
        C = i.capitalize()
        D+= C + ' '
    D+=B[-1].capitalize()


    idx = indices[D]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the most similar movies
    sim_scores = sim_scores[0:7]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    return (df['Title'].iloc[movie_indices].unique(), df['Genre'].iloc[movie_indices].unique(),  df['Rating'].iloc[movie_indices].unique())
#######################################################################################

# defining an instance of the flask app
app = Flask(__name__)

#routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search',methods = ['POST','GET'])
def searcher():
    try:
        inpu = request.form['text']
        movies,genres,rating = get_recommendations(inpu)
        
        return render_template('index.html',movies = zip(movies,genres,rating))
    except:
        return render_template('err.html')


#running the application
if __name__ == '__main__':
    app.run(debug=True)