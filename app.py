from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import recommend_from_offer

app = Flask(__name__)
CORS(app)  # pour autoriser Flutter à communiquer

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()

    # Récupération des données
    experience = data.get("experience", "")
    skills = data.get("skills", [])
    count = data.get("count", 3)

    # Création du texte d'offre unifié
    offer_text = experience + " " + " ".join(skills)
    print(offer_text)

    # Appel du moteur de recommandation
    profiles = recommend_from_offer(offer_text, top_n=count)


    return jsonify({"profiles": profiles})


if __name__ == '__main__':
    app.run(debug=True, port=5051)