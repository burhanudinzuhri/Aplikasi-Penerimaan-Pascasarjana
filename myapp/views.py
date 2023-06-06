from django.shortcuts import render

# Create your views here.
from .models import MyData
from django.http import HttpResponse
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd
import numpy as np
import pickle
import os
from pickle import load
from tensorflow import keras
from keras.models import load_model

# Create your views here.
def home(request):
    return render(request,'index.html')

def display(request):
    return render(request,'input.html')

def save(request):
    if request.method=="POST":
        s=MyData()
        s.TOFEL=request.POST.get('ht_tofel')
        s.GRE=request.POST.get('ht_gre')
        s.UNI_rating=request.POST.get('ht_Uni_rating')
        s.SOP=request.POST.get('ht_sop')
        s.LOR=request.POST.get('ht_lor')
        s.CGPA=request.POST.get('ht_cgpa')
        s.Research_Ex=request.POST.get('ht_research')


        #read the data in data frame
        data=[[s.GRE,s.TOFEL,s.UNI_rating,s.SOP,s.LOR,s.CGPA,s.Research_Ex]]
        newx=pd.DataFrame(data,columns=["GRE","TOFEL","UNI_rating","SOP","LOR","CGPA","Research_Ex"])

        #loading model and data using  pickle
        # filename = "/Admission/myapp/data/dmission_model.sav"

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(BASE_DIR)

        filename = BASE_DIR + '/myapp/data/model_cnn_lstm.h5'
        filename1 = BASE_DIR + '/myapp/data/scaler_ds.pkl'

 
        # load the model from disk

        model = load_model(filename)
        x = pickle.load(open(filename1,"rb"))

        #apply minMax scaler for scaling the data
        # scalerX = MinMaxScaler(feature_range=(0, 1))
        # x[x.columns] = scalerX.fit(x)
        
        newx[newx.columns] = x.transform(newx[newx.columns])

        #here we predict the score on new data
        y_predict = model.predict(newx)
        
        s.Chance_of_Admit = y_predict
        s.save()
        
        #return HttpResponse('Record submitted successfully')
        return render(request,'output.html', {'score':y_predict})