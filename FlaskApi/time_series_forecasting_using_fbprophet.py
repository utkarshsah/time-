# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:20:13 2021

@author: kalas
"""
from flask import Flask,render_template,redirect,request
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import itertools
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from random import randint
import plotly.graph_objs as go
import plotly.offline as py

import base64
from io import BytesIO
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "static/img/"
@app.route('/')
def hello():
    return render_template("step1.html")




@app.route("/home")
def home():
    return redirect('/')


@app.route('/',methods=['POST'])
def submit_data():
  try:
    print("\n>>> Form submitted, starting forecast...")
    f=request.files['userfile']
    s1=request.form['query1']
    s2=request.form['query2']  
    t=int(request.form['query3'])
    s4=request.form['query4']
    freq_aliases = {
        'M': 'ME',
        'Y': 'YE',
    }
    s4 = freq_aliases.get(s4, s4)
    df=pd.read_csv(f)
    print(f">>> CSV loaded: {df.shape[0]} rows, columns: {list(df.columns)}")
    
    #Prophet
    df = df.rename(columns={s2: 'y', s1:'ds'})
    df['y_orig'] = df['y'] # to save a copy of the original data..you'll see why shortly. 
    df['y'] = np.log(df['y'])
    print(">>> Fitting Prophet model (this may take a minute on first run)...")
    model = Prophet() #instantiate Prophet
    model.fit(df)
    print(">>> Model fitted successfully!")

    
    
    ''' 'year': 'A',
        'quarter': 'Q',
                'month': 'M',
                'day': 'D',
                'hour': 'H',
                'minute': 'T',
                'second': 'S',
                'millisecond': 'L',
                'microsecond': 'U',
                'nanosecond': 'N'}
        '''
        
        
    future_data = model.make_future_dataframe(periods=t, freq = s4)
    print(f">>> Future dataframe created: {len(future_data)} rows, freq={s4}")
    forecast_data = model.predict(future_data)
    print(">>> Prediction complete!")

    
    forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
    
    model.plot(forecast_data) 
    model.plot_components(forecast_data)
    forecast_data_orig = forecast_data # make sure we save the original forecast data
    forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
    model.plot(forecast_data_orig)
    df['y_log']=df['y'] #copy the log-transformed data to another column
    df['y']=df['y_orig']
    final_df = pd.DataFrame(forecast_data_orig)
    
    final_df_1=final_df[['ds','yhat']].tail(t)
    final_df_1 = final_df_1.rename(columns={'yhat': 'Sales', 'ds':'Month'})
    

    #rmse = mean_squared_error(df["y_orig"].iloc[24:], final_df['yhat'].iloc[24:36])**0.5
    #print('Test MSE: %.3f' % rmse)
                
      
    fig,ax=plt.subplots(nrows=1, ncols=1)
    ax.plot(df["y_orig"],label="Actual")
    ax.plot(final_df["yhat"],label="Predicted")
    ax.legend()

    #plt.xticks(rotation=90)
    #plt.show()
    image_buffer = BytesIO()
    fig.savefig(image_buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    image_buffer.seek(0)
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    full_filename = f"data:image/png;base64,{image_base64}"
    
    print(">>> Chart generated, sending response!\n")
    return render_template('step1.html',user_image = full_filename,tables=[final_df_1.to_html(classes='forecast')],titles=['na','forecast'],query1 = request.form['query1'],query2 = request.form['query2'],query3 = request.form['query3'], query4 = request.form['query4'])
  except Exception as e:
    import traceback
    traceback.print_exc()
    return f"<h2>Error</h2><pre>{traceback.format_exc()}</pre>", 500
     
'''
    import plotly.graph_objs as go
    import plotly.offline as py
    #Plot predicted and actual line graph with X=dates, Y=Outbound
    actual_chart = go.Scatter(y=df["y_orig"], name= 'Actual')
    predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
    predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
    predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')
    #py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower])
    py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower], filename = 'templates/' +'filename.html', auto_open=False, image_width=200, image_height=200)
    
'''
    #return render_template('step1.html',user_image = full_filename,tables=[final_df_1.to_html(classes='forecast')],titles=['na','forecast'],query1 = request.form['query1'],query2 = request.form['query2'],query3 = request.form['query3'], query4 = request.form['query4'])
    
   
if __name__ =="__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask on port {port} with debug=True")
    app.run(host="0.0.0.0", port=port, debug=True)
    
