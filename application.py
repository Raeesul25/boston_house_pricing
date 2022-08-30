import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# starting point ofapplication
app = Flask(__name__, template_folder='template')

# load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
# load scaler 
scaler = pickle.load(open('scaling.pkl', 'rb'))

# go to my home page
@app.route('/')
def home():
    return render_template('home.html')

# predict api
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__=="__main__":
    app.run(debug=True)