from flask import render_template
from flask import request
from boulderapp import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import numpy as np
from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

user = 'nicholasgriffin' #add your username here (same as previous postgreSQL)                      
host = 'localhost'
dbname = 'birth_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Nick' },
       )

@app.route('/db')
def birth_page():
    sql_query = """                                                                       
                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean';          
                """
    query_results = pd.read_sql_query(sql_query,con)
    births = ""
    for i in range(0,10):
        births += query_results.iloc[i]['birth_month']
        births += "<br>"
    return births

@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = """
  	       SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,con)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'],birth_month=query_results.iloc[i]['birth_month']))
    return render_template('starter-template.html',births=births)

@app.route('/input')
def boulderjoshua_input():
    return render_template("input.html")

@app.route('/output')
def boulderjoshua_output():
    #pull 'birth_month' from input field and store it
    dinput = request.args.get('input')
    forecastabbrev10 = pd.read_csv("~/Documents/campgroundpredictions_twoyears.csv")
    az = forecastabbrev10.loc[forecastabbrev10['date'] == dinput]
    az2=az[['belle_yhat','br_yhat','cw_yhat','hv_yhat','ic_yhat','jr_yhat','ryan_yhat','wt_yhat']]
    az3=pd.DataFrame(az2)
    az3.columns=['Belle','Black Rock','Cottonwood','Hidden Valley','Indian Cove','Jumbo Rocks','Ryan','White Tank']
    az3=pd.melt(az2)
    az3=az3.sort_values('value')
    az3=az3.reset_index()
    bestcamp = (az3['variable'][0])
    bcocc = (az3['value'][0])
    print(bestcamp)
    print(bcocc)
    bestcamp2 = (az3['variable'][1])
    bcocc2 = (az3['value'][1])
    bestcamp3 = (az3['variable'][2])
    bcocc3 = (az3['value'][2])
    print(bestcamp2)
    print(bcocc2)
    print(bestcamp3)
    print(bcocc3)
    
    #just select the Cesareans  from the birth dtabase for the month that the user inputs
    #query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
    #print (query)
    #query_results=pd.read_sql_query(query,con)
    #print (query_results)
    #births = []
    #for i in range(0,query_results.shape[0]):
    #    births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    #    the_result = ModelIt(patient,births)
    return render_template("output.html", bestcamp = bestcamp, bcocc = bcocc,  bestcamp2 = bestcamp2, bcocc2 = bcocc2,  bestcamp3 = bestcamp3, bcocc3 = bcocc3)
