from flask import Flask, request, url_for, redirect, render_template, jsonify
import json
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import glpk
#import scipy
import math
from datetime import datetime
import pylab as pl
#from random import shuffle
import Configuration
from Configuration import configuration
import webbrowser

users = Configuration.configuration.users
Patientuser = Configuration.configuration.users1
AdminFilename = Configuration.configuration.AdminFilename
PatientFilename = Configuration.configuration.PatientFilename
patientimg = Configuration.configuration.fig


print("Patient User is:",Patientuser.values())



def user_add():
    with open(PatientFilename) as file:
        adminData = json.load(file)

        return adminData
adminData = user_add()

def RipDict():
    for ii in range(len(adminData)):
        PatientDictName = "Patient ID " + str(ii)
        

RipDict()

#ConverAdminDatatoDict(adminData)





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





#adminData = user_add()
PatientId = CreateUserdata()
NumberofElements = len(adminData)




# Create the FlaskApp
app = Flask(__name__)
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
    print("username is", username)
    if username in users and users[username] == password:
        return redirect(url_for('admin'))
    elif username in Patientuser and Patientuser[username] == password:
       # return render_template('PatientPage.html',patientname=password)
       return redirect(url_for('PatientScreen'))
    else:
        error = 'Invalid username or password'
        return render_template('index.html', error=error)

@app.route('/admin')
def admin():
    return render_template('JSONread.html', data=adminData)

@app.route('/PatientScreen')
def PatientScreen():
    return render_template('PatientPage1.html')



@app.route('/ManageModelPatient', methods = ['POST', 'GET'])
def ManageModelPatient():
    #me
    #patientdetail =
    patientname = request.form['Patientname']
    if patientname in Patientuser.values():
        return render_template('ControlManageSelect.html', patientname = patientname)
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

    print(PatientId)
    item = PatientId
    #print(filename)
    #image = patientname.index()
    #image = patientimg[patientname]['measure']

    if request.form['operation'] == 'Control':
        filename = './assets/img/management/' + str(patientname) + '.jpg'
        return render_template('bgl.html', image = filename , patientname=patientname , item = item)
    if request.form['operation'] == 'Model':

        filename = './assets/img/model/' + str(patientname) + '.jpeg'
        return render_template('bgl.html', image=filename, patientname=patientname, item=item)
    else:
        request.form['operation'] == 'ClinicalTrials'
        filename = './assets/img/measure/' + str(patientname) + '.JPG'
        return render_template('bgl.html', image=filename, patientname=patientname, item=item)











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
    

