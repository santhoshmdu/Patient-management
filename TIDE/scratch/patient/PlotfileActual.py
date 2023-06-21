from flask import Flask,render_template,url_for,request,redirect, make_response
import random
import json
#from time import time
from random import random
from flask import Flask, render_template, make_response
from time import time
import numpy as np
import pandas as pd




app = Flask(__name__)

t = np.arange(0,1440,8)
Index = 0
df = pd.read_csv('simdata.csv')
G = list(df.iloc[:,0])
F = list(df.iloc[:,1])
I = list(df.iloc[:,2])
lenofG = len(G)
print(lenofG)


def Increment():
    global Index
    Index = Index + 1
    return Index



def gAndIval(Index):
    gval = G[Index]
    Fval = F[Index]
    Ival = I[Index]
    return gval,Fval,Ival

@app.route('/', methods=["GET", "POST"])
def main():

    return render_template('plotlivedata22.html')

@app.route('/bgl', methods=["GET", "POST"])
def data():
    Index = Increment()
    while (Index <= lenofG - 1):
        gval,Fval,Ival = gAndIval(Index)
        data = [Index,gval,Fval,Ival]
        response = make_response(json.dumps(data))
        response.content_type = 'application/json'
        return response

if __name__ == "__main__":
    app.run(debug=True)
