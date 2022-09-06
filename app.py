from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model=pickle.load(open('random_forest_regression_model.pkl','rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('covid_page.html')
standard_to=StandardScaler()
@app.route("/predict",methods=['POST'])
def predict():
    if request.method=='POST':
        BP=int(request.form['Breathing Problem'])
        DC=int(request.form['Dry Cough'])
        ST=int(request.form['Sore throat'])
        RN=int(request.form['Running Nose'])
        HD=int(request.form['Heart Disease'])
        FT=int(request.form['Fatigue'])
        GT=int(request.form['Gastrointestinal'])
        AT=int(request.form['Abroad travel'])
        CP=int(request.form['Contact with COVID patient'])
        ALG=int(request.form['Attended Large Gathering'])
        VPEP=int(request.form['Visited Public Exposed Places'])
        FWPEP=int(request.form['Family working in Public Exposed Places'])
        prediction=model.predict([[BP,DC,ST,RN,HD,FT,GT,AT,CP,ALG,VPEP,FWPEP]])
        output=prediction[0]
        if output==0:
            return render_template('covid_page.html',prediction_text="You are safe!")
        else:
            return render_template('covid_page.html',prediction_text="Your are affected with covid,please take rest.Stay safe and healthy!")
    else:
        return render_template('covid_page.html')

            
if  __name__=="__main__":
    app.run(debug=True)
