#imports Flask
from flask import Flask
from flask.globals import request
from flask.wrappers import Response
#imports bibliothèques manipulation données
import joblib
import pandas as pd

#chargement modèle
model = joblib.load('modele.sav')

#chargement dataset
clients = pd.read_csv('Data/sample.csv')
clients.set_index('SK_ID_CURR', inplace = True)

#on initialise l'API
app = Flask(__name__)

#on définit une route, url, avec l'ID du client à prédire
@app.route('/predict/<id_client>')
def predict(id_client:int):
    
    #récupérer l'index de l'ID
    id_client = int(id_client)
    index = clients.index.get_loc(id_client)

    #on récupère les features du client
    features = clients.iloc[index]

    #on créer un disctionnaire pour la prédiction
    response =  {}
    response['predictions'] = model.predict_proba([features])[0,1].tolist()

    #on retourne le dictionnaire avec la prédiction
    return response

if __name__ == "__main__":
    app.run(debug=True)