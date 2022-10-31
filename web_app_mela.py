# -*- coding: utf-8 -*-
# Author Claudia Cozzolino https://github.com/coclab
# Credits to https://towardsdatascience.com/building-a-web-application-to-deploy-machine-learning-models-e224269c1331




# import libraries for web app
#import os
from flask import Flask, request, redirect, url_for, render_template


# import libraries for data manipulation and model
import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow import keras  # tf.keras

#set seed for reproducibility of results
np.random.seed(42) 
tf.random.set_seed(42)

# load model
DNNs_fit = keras.models.load_model('./DNNs trained.h5')


# create istance of Flask app
app = Flask(__name__)


# app settings

'''
User will interact with my main page in two ways: 
1. Load the web page from the server into his browser -> GET
2. Upload data completing the form and send them to the server -> POST
'''


@app.route('/', methods=['GET', 'POST']) #default home page
def main_page():
    if request.method == 'POST':
        
        # read form input and save in global var
        global X, age, mitotic, gender, histology, T, N, M, ulceration, regression, growth, til, site
        
        # prepare empty df where store input (as X1 names order)
        column_names = ['Male', 'age', 'Mitotic count', 'Histology Desmoplastic melanoma',
               'Histology Lentigo maligna', 'Histology Malignant melanoma',
               'Histology Melanoma arising from blue naevus',
               'Histology Nodular melanoma', 'Histology Spitzoid melanoma',
               'Histology Superficial spreading melanoma', 'T1 stage', 'T2 stage',
               'T3 stage', 'T4 stage', 'TX stage', 'N0 stage', 'N1a stage', 'N1b stage', 'N1c stage',
               'N2a stage', 'N2b stage', 'N2c stage', 'N3 stage', 'N3c stage',
               'M1 stage', 'Ulceration Present', 'Tumor regression Present',
               'Growth pattern Vertical', 'TIL Present', 'Tumor site Upper limb',
               'Tumor site Head', 'Tumor site Hands/Feet', 'Tumor site Trunk']
    
        X = pd.DataFrame(columns = column_names) # empty
        
        
        age = float(request.form.get('age'))         # numeric cast from string
        mitotic = float(request.form.get('mitotic'))

        # save form input
            
        gender = request.form.get('gender') 
        histology = request.form.get('histology') 
        T = request.form.get('T') 
        N = request.form.get('N')
        M = request.form.get('M')
        ulceration = request.form.get('ulceration') 
        regression = request.form.get('regression') 
        growth = request.form.get('growth') 
        til = request.form.get('til') 
        site = request.form.get('site') 
        
        
        
        # encode form input in model input format
                
        X['Male'] = [1*(gender == 'Male')]
        X['age'] = [age]
        X['Mitotic count'] = [mitotic] 
        
        X['Histology Desmoplastic melanoma'] = [1*(histology == 'Desmoplastic')]  
        X['Histology Lentigo maligna'] = [1*(histology == 'Lentigo maligna')]
        X['Histology Malignant melanoma'] = [1*(histology == 'Malignant')]
        X['Histology Melanoma arising from blue naevus'] = [1*(histology == 'Arising from blue naevus')]
        X['Histology Nodular melanoma'] = [1*(histology == 'Nodular')]
        X['Histology Spitzoid melanoma'] = [1*(histology == 'Spitzoid')]
        X['Histology Superficial spreading melanoma'] = [1*(histology == 'Superficial spreading')]
        
        
      
        X['TX stage'] = [1*(T == 'TX')]  
        X['T1 stage'] = [1*(T == 'T1')]  
        X['T2 stage'] = [1*(T == 'T2')]
        X['T3 stage'] = [1*(T == 'T3')]
        X['T4 stage'] = [1*(T == 'T4')]
        
        X['N0 stage'] = [1*(N == 'N0')]   
        X['N1a stage'] = [1*(N == 'N1a')]   
        X['N1b stage'] = [1*(N == 'N1b')]
        X['N1c stage'] = [1*(N == 'N1c')]
        X['N2a stage'] = [1*(N == 'N2a')]
        X['N2b stage'] = [1*(N == 'N2b')]
        X['N2c stage'] = [1*(N == 'N2c')]
        X['N3 stage'] = [1*(N == 'N3')]
        X['N3c stage'] = [1*(N == 'N3c')]
        
        X['M1 stage'] = [1*(M == 'M1')]    
        
        X['Ulceration Present'] = [1*(ulceration == 'Present')]
        X['Tumor regression Present'] = [1*(regression == 'Present')]
        X['Growth pattern Vertical'] = [1*(growth == 'Vertical')]
        X['TIL Present'] = [1*(til == 'Present')]
        
        X['Tumor site Upper limb'] = [1*(site == 'Upper limb')]
        X['Tumor site Head'] = [1*(site == 'Head')]
        X['Tumor site Hands/Feet'] = [1*(site == 'Hands or feet')]
        X['Tumor site Trunk'] = [1*(site == 'Trunk')]
        
       

        # and go to prediction page 
        return redirect(url_for('prediction'))
        
        
        
    return render_template('index.html')




# perform prediction and pass results to prediction page
@app.route('/prediction')
   
def prediction():
    
    global probabilities, predictions
    
    
    
    # Using the model, predict the probabilities to survive in 3 years from diagnosis
    probability = DNNs_fit.predict(X.iloc[0,:].values.reshape(1, -1))[0][0]
    probabilities = np.array([1-probability, probability])
    
    
    print(probabilities)
      
    # Find the label and the probability of the top three most probable classes, and put them under predictions
    number_to_class = ['Yes', 'No'] # survive?
    index = np.argsort(1-probabilities) # descending
    predictions = {
      "class1":number_to_class[index[0]],
      "class2":number_to_class[index[1]],
      "prob1": round(100 * probabilities[index[0]], 2),
      "prob2": round(100 * probabilities[index[1]], 2)
    }
    

      
    # Load the template of my webpage predict.html and give them the predictions in Step 3.
    return render_template('predict.html', predictions=predictions)










if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80) #when in server
    #app.run(debug=False) #when run in local host
    
    
    

    
