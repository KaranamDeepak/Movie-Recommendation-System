from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)

CORS(app)

# load model files
movies = pickle.load(
    open('movie_list.pkl', 'rb')
)

similarity = pickle.load(
    open('similarity.pkl', 'rb')
)

# home route
@app.route("/")
def home():
    return "Movie Recommendation API Running"

# recommendation route
@app.route("/recommend", methods=["POST"])
def recommend():

    data = request.json

    movie = data["movie"]

    movie_index = movies[
        movies['title'] == movie
    ].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_list:

        recommendations.append(
            movies.iloc[i[0]].title
        )

    return jsonify({
        "recommendations": recommendations
    })

# run flask app
if __name__ == "__main__":
    app.run(debug=True)