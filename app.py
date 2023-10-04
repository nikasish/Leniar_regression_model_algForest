from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

scaler_file = pickle.load(open('models/scaler.pkl', 'rb'))
model_file = pickle.load(open('models/reg.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    if request.method == 'POST':
        Temperature = request.form.get('Temperature')
        RH = request.form.get('RH')
        Ws = request.form.get('Ws')
        Rain = request.form.get('Rain')
        FFMC = request.form.get('FFMC')
        DMC = request.form.get('DMC')
        DC = request.form.get('DC')
        ISI = request.form.get('ISI')
        BUI = request.form.get('BUI')
        Classes = request.form.get('Classes')
        
        new_scaled_data = scaler_file.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes]])
        
        result = model_file.predict(new_scaled_data)
        return render_template('new.html', result = result)
    
    return render_template('new.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')