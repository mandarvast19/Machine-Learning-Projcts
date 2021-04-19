import numpy as np
import pandas as pd
from flask import Flask,request,render_template,redirect,url_for
import pickle

app = Flask(__name__)
app.config['DEBUG']=True

@app.route('/',methods=['GET'])
def home():
    return render_template('flight_fare.html')

model = pickle.load(open('rf_flight.pkl','rb'))
    
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        airline = request.form['Airline']
        source = request.form['Source']
        destination = request.form['Destination']
        total_stops = int(request.form['Total_Stops'])
        departure_date = request.form['Departure_Date']
        arrival_date = request.form['Arrival_Date']

        #Journey
        Journey_day = int(pd.to_datetime(departure_date).day)
        Journey_month = int(pd.to_datetime(departure_date).month)

        #departure
        Dep_Time_hour = int(pd.to_datetime(departure_date).hour)
        Dep_Time_minute = int(pd.to_datetime(departure_date).minute)

        #arrival
        Arrival_Time_hour = int(pd.to_datetime(arrival_date).hour)
        Arrival_Time_minute = int(pd.to_datetime(arrival_date).minute)

        #duration in mins

        arrival_date = pd.to_datetime(arrival_date)
        departure_date = pd.to_datetime(departure_date)
        duration = abs(arrival_date-departure_date)
        Duration_in_mins = duration.total_seconds()/60

        #Airline
        a = {'Air Asia':0,'Air India':1,'GoAir':2,'IndiGo':3,'Jet Airways':4, 'Jet Airways Business':5,
            'Multiple carriers':6,'Multiple carriers Premium economy':7, 'SpiceJet':8,
            'Trujet':9, 'Vistara':10, 'Vistara Premium economy':11}
        airline_array = np.zeros(12)
        index = a[airline]
        airline_array[index] = 1
        
        #Source
        source_dict = {'Banglore':0,'Chennai':1,'Delhi':2,'Kolkata':3,'Mumbai':4}
        source_array = np.zeros(5)
        index = source_dict[source]
        source_array[index] = 1
        
        #Destination
        destination_dict = {'Banglore':0,'Cochin':1,'Delhi':2,'Hyderabad':3,'Kolkata':4,'New Delhi':5}
        destination_array = np.zeros(6)
        index = destination_dict[destination]
        destination_array[index] = 1

        final_array = [Duration_in_mins,total_stops,Journey_day,Journey_month,Dep_Time_hour,Dep_Time_minute,
               Arrival_Time_hour,Arrival_Time_minute]+list(airline_array)+list(source_array)+list(destination_array)
        data = np.array(final_array)
        data = data.reshape(1,-1)

        price = model.predict(data)
        price = round(price[0],2)
        return render_template('result.html',predicted_price='₹ {}'.format(price))


if __name__=="__main__":
    app.run()
     
