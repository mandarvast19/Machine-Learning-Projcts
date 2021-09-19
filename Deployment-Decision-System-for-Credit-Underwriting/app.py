import numpy as np
import pandas as pd
from flask import Flask,request,render_template,redirect,url_for
import pickle
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('credit_underwriting.html')

model = pickle.load(open('logreg.pkl','rb'))
scaler = joblib.load('standard_scaler.pkl')
    
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        credit = request.form['credit_amount']
        income = request.form["income_amount"]
        goods = request.form['goods_price']

        gender = request.form['flexRadioDefault']
        age = request.form['age']
        education_type = request.form['education_type']

        property = request.form['flexRadioDefault2']
        car = request.form['flexRadioDefault3']
        ext_score = request.form['ext_score']

        occupation_type = request.form['occupation_type']
        income_type = request.form['income_type']


        ## preprocessing

        to_scale = np.array([[ 0.42687192,  1.78998998,  0.61369265,  1.66868837, -1.25795128,
        0.16361679, -0.47824207, -1.08277836, -1.78887278, -0.31212956,
        0.57217871,  0.00313769, -0.17713951, -0.17163284,  0.20763664,
       -0.08179008, -0.06920231, -0.18559639, -0.31800391, -0.3640824 ,
       -1.13711806, -1.36685867]])
       
        to_scale[0,1]=float(credit)
        to_scale[0,0]=float(income)
        to_scale[0,3]=float(goods)
        to_scale[0,5]=float(age)
        to_scale[0,10]=float(ext_score)

        to_scale = scaler.transform(to_scale)

        credit = to_scale[0,1]
        income = to_scale[0,0]
        goods = to_scale[0,3]
        age = to_scale[0,5]
        ext_score = to_scale[0,10]

        # Gender
        gender_array= np.zeros(3)
        gender = int(gender)
        gender_array[gender] = 1

        # Education Type
        education_array = np.zeros(5)
        education_type = int(education_type)
        education_array[education_type] = 1

        # Property and car
        property_array = np.zeros(2)
        property = int(property)
        property_array[property] = 1

        car_array = np.zeros(2)
        car = int(car)
        car_array[car] = 1

        # Occupation Type
        occupation_array = np.zeros(11)
        occupation_type = int(occupation_type)
        occupation_array[occupation_type] = 1

        # Income Type
        income_array = np.zeros(7)
        income_type = int(income_type)
        income_array[income_type] = 1

        input_array = [credit,income,goods]+list(gender_array)+[age]+list(education_array)+list(property_array)+list(car_array)+[ext_score]+list(occupation_array)+list(income_array)
        input_array = np.array(input_array)
        input_array = input_array.reshape(1,-1)

        print(input_array)
         
        prediction = model.predict(input_array)
        if prediction==0:
            results = "Not a Defaulter"
            color = "text-success"
        else:
            results = "a Defaulter"
            color = "text-danger"
        return render_template('result.html',predicted_result=results, color=color)
        # predicted_result='{0}'.format(results)



if __name__=="__main__":
    app.run(debug=True)