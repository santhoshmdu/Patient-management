from flask import Flask, request, url_for, redirect, render_template, jsonify ,make_response
import json
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import glpk
#import scipy
from datetime import datetime
import pylab as pl
#from random import shuffle
import Configuration
from Configuration import configuration
import webbrowser
import random
from random import random
from time import time

users = Configuration.configuration.users
Patientuser = Configuration.configuration.users1
AdminFilename = Configuration.configuration.AdminFilename
PatientFilename = Configuration.configuration.PatientFilename
patientimg = Configuration.configuration.fig


def user_add():
    with open(PatientFilename) as file:
        adminData = json.load(file)

        return adminData
adminData = user_add()

def ConverAdminDatatoDict(adminData):
    it = iter(adminData)
    AdminDict = dict(zip(it,it))
    print(AdminDict)
    return AdminDict

def CreateUserdata():
    for ii in range(len(adminData)):
        PatientId = adminData[0]
        print("Patient id is", PatientId)
        return PatientId

PatientId = CreateUserdata()
NumberofElements = len(adminData)




# Create the FlaskApp
app = Flask(__name__)






@app.route('/admin')
def admin():
    return render_template('JSONread.html', data=adminData)



@app.route('/patientScreen')
def patientScreen():
    password = request.args.get('password')
    patient = password
    matching_patients = [patient for patient in adminData if patient["S.No"] == password]
    
    return render_template('JSONread.html', data=matching_patients )


patientname = None
t = None
Index = None
df = None
G = None
F = None
I = None
lenofg = None

@app.route('/ManageModelPatient', methods=['POST', 'GET'])
def ManageModelPatient():

    global patientname, t, Index, df, G, F, I, lenofg   # Declare the variables as global
    patientname = request.form['Patientname']
    mode = request.form.get('mode')  # Retrieve the selected mode
    
    if patientname in Patientuser.values():
         
        if mode == '1':  # Check if mode is set to '1' (Real-time)
            t, Index, df, G, F, I, lenofg = new(patientname)
            return redirect(url_for('main'))  # Redirect to the '/real' route
        
        # Handle other modes or conditions if needed
       
        return render_template('ControlManageSelect.html', patientname=patientname )

    else:
        return "Patient Not Available in Database"



@app.route('/ManageModelPatient1/<patientname>', methods = ['POST', 'GET'])
def ManageModelPatient1():
    patientname = request.form['Patientname']
    if patientname in Patientuser.values():
        #detail = PatientId
        #image = PatientId['Sex']
        return render_template('ControlManageSelect.html',  patientname = patientname  )
    else:
        return "Patient Not Available in Database"


@app.route('/ManageModelPatientOperation/<patientname>', methods = ['POST', 'GET'])
def ManageModelPatientOperation(patientname):
    matching_patients = [patient for patient in adminData if patient["S.No"] == patientname]
    item = matching_patients
    #print(item)
    #print(filename)
    #image = patientname.index()
    #image = patientimg[patientname]['measure']
    #print(patientname)
    if request.form['operation'] == 'Control':
        print(item[0]['S.No'])
        filename = './assets/img/management/' + str(patientname) + '.jpg'
        return render_template('bgl.html', image = filename , item = item ,patientname=patientname)
    if request.form['operation'] == 'Model':

        filename = './assets/img/model/' + str(patientname) + '.jpeg'
        return render_template('bgl.html', image=filename, item=item , patientname=patientname)
    else:
        request.form['operation'] == 'ClinicalTrials'
        filename = './assets/img/measure/' + str(patientname) + '.JPG'
        return render_template('bgl.html', image=filename,  item=item ,patientname=patientname)





# Render the homepage
@app.route('/')
def home():
    return render_template(Configuration.configuration.file1)
# Render the Index Page
@app.route('/index')
def Index():
    return render_template('index.html')
#Login page
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username in users and users[username] == password:
        return redirect(url_for('admin'))
    elif username in Patientuser and Patientuser[username] == password:
        return redirect(url_for('patientScreen',password=password))
        #return render_template('PatientPage.html',patientname=username)

    else:
        error = 'Invalid username or password'
        return render_template('index.html', error=error)


#@app.route('/adminLogin')
#def ManagePatient(username,password):


#@app.route('/user_add')
#def user_add(u):
        #   with open(AdminFilename) as file:
        #adminData = json.load(file)
        #patientName = adminData["PatientP1"]
        #ManageFile = patientName["ManageFile"]
        #return render_template('patientread.html',data =adminData)




    #return render_template('patientread.html', data = adminData)
def new(patientname):
    t = np.arange(0,1440,8)
    Index = 0
    csv = str(patientname)+".csv"
    df = pd.read_csv(csv)
    G = list(df.iloc[:,0])
    F = list(df.iloc[:,1])
    I = list(df.iloc[:,2])
    lenofg = len(G)
    print(lenofg)
    return t ,Index, df ,G ,F ,I , lenofg




def Increment():
    global Index
    Index = Index + 1
    return Index



def gAndIval(Index):
    gval = G[Index]
    Fval = F[Index]
    Ival = I[Index]
    return gval,Fval,Ival

@app.route('/real', methods=["GET", "POST"])
def main():
    
    print(patientname)
    return render_template('plotlivedata22.html')

@app.route('/bgl', methods=["GET", "POST"])
def data():
    Index = Increment()
    while (Index <= lenofg - 1):
        gval,Fval,Ival = gAndIval(Index)
        data = [Index,gval,Fval,Ival]
        response = make_response(json.dumps(data))
        response.content_type = 'application/json'
        return response

if __name__ == '__main__':
    port = Configuration.configuration.port
    webbrowser.open(port)
    app.run(debug=True)























@app.route('/patient/<patientid>/manage')
def manage(patientid):
    Dt = pd.read_csv('simdata.csv')

    Df = Dt.copy()


    bgl = Df.iloc[:, 0]
    F = Df.iloc[:, 2]
    I = Df.iloc[:, 1]
    #bgl_r = Df.iloc[:, 2]
    Gg = []
    Ig = []
    Glu = bgl

    n=len(Glu)
    lam=0.98
    delta=1e2
    P=delta*np.identity(3)
    #P1 = pd.DataFrame(P)
    w1 = np.matrix(np.zeros(3))
    w = w1.T
    Wp = []
    e=[]
    y1 = []
    # initial condition
    Gk = Glu[0]
    N = 4
    # Constraints
    Gmax = 180 * np.ones((N, 1))
    Gmin = 100 * np.ones((N, 1))
    for i in range(1, n-1):
        u1 = np.matrix([I[i], F[i], Glu[i]])
        u12 = np.array(u1)
        u = u1.T        #,Glu[i+2]]
        phi = np.matrix(u.T*P)
        k = phi.T/(lam+phi*u)
        y = w.T*u
        yp = pd.DataFrame(y)
        y1.append(yp[0])
        #print(i, u)
        e = Glu[i+1]-y
        w = w+k*e
        P = (P-k*phi)/lam
        b = w[0]
        c1 = w[1]
        a = w[2]
        Wp.append(w)
        Cal_A = np.zeros((N, 1))
        for ct in range(0, N, 1):
            Cal_A[ct] = a ** (ct + 1)
        Cal_B = np.zeros((N, N))
        Cal_B[0, 0] = b
        for k in range(1, N, 1):
            Cal_B[k, :] = a * Cal_B[k - 1, :]
            Cal_B[k, k] = b
        Cal_B = np.asmatrix(Cal_B)
        Cal_C = np.zeros((N, N))
        Cal_C[0, 0] = c1
        for kt in range(1, N, 1):
            Cal_C[kt, :] = a * Cal_C[kt - 1, :]
            Cal_C[kt, kt] = c1
        Cal_C = np.asmatrix(Cal_C)


    t = np.arange(0,1440,8)

    n = len(Glu)
    import matplotlib as mpl
    mpl.style.use("default")
    fig,ax = plt.subplots(3,figsize=(8,8))
    ax[0].plot(t,Glu,'-',label = "ODE simulated data")
    ax[0].legend(loc = 'upper left')
    ax[0].plot(t[4:n],y1[2:n],'--',label = "Linear model data")
    ax[0].set_ylabel("BGL (mg/dl)")
    ax[0].legend(loc = 'upper left')
    ax[0].set_title('Blood glucose level',fontsize = 10)
    ax[1].plot(t,(F/(100)),label="CHO present in food")
    ax[1].set_ylabel("Grams")
    ax[0].legend(loc = 'upper left')
    ax[1].set_title('CHO present in food')
    ax[2].plot(t,(I/100),label="Insulin infusion")
    ax[2].set_xlabel("Time in minutes")
    ax[2].set_ylabel("Units")
    ax[0].legend(loc = 'upper left')
    ax[2].set_title('Insulin infusion')
    fig.tight_layout()

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    encoded_img = base64.b64encode(img_data.getvalue()).decode()

    # Return the base64-encoded image data as JSON response
    return render_template('manage.html', image=encoded_img)

    #plt.figure(1)
    #plt.plot(Glu)
    #plt.plot(y1[2:n])
    #plt.plot(Gg)
    #plt.figure(2)
    #plt.plot(Ig)
    #plt.plot(Ig)
    #plt.plot(t[2:n], y1[:n], '-', c = "k", label = "Model fit", marker = 'o') #rotation=30)
    #plt.plot(t[2:n], Glu[2:n], '--', c="r", label = "Measurements", marker = 'o')
    

