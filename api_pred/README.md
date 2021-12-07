# API Pred
This is the source code for the API ["API Pred"](https://apipred.herokuapp.com/predict/248515), hosted on Heroku. <br>
This API is used in [this Dashboard](https://github.com/Boudh/Dash). <br>
The goal of this API is to returns the probability that a customer of the bank does not repay his loan. 

__Files :__
- api.py : the main file of the Flask API
- modele.sav : the model trained, a SGD Classifier
- Data/sample.csv : the CSV file containing data sample from bank customers
- Procfile and requirement.txt : files needed to init and configure the server hosting the API

## How it works ? 
The API take in parameters the ID of a bank customer from sample.csv. <br>
Then the API finds the bank customers informations and the model returns the probability that a client will default on a loan. <br>
You can try the API by specifying a customer ID in the url : /predict/__id_client__
