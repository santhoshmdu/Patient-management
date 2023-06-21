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
    if username in users and users[username] == password:
        return redirect(url_for('user_add'))
    elif username in Patientuser and Patientuser[username] == password:
        return 'PatientPage'
    else:
        error = 'Invalid username or password'
        return render_template('index.html', error=error)


@app.route('/adminLogin')
def ManagePatient(username,password):
    with open(AdminFilename) as file:
        adminData = json.load(file)
        patientName =


#@app.route('/user_add')
#def user_add(u):
        #   with open(AdminFilename) as file:
        #adminData = json.load(file)
        #patientName = adminData["PatientP1"]
        #ManageFile = patientName["ManageFile"]
        #return render_template('patientread.html',data =adminData)


    #return render_template('patientread.html', data = adminData)



if __name__ == '__main__':
    port = Configuration.configuration.port
    webbrowser.open(port)
    app.run(debug=True)








@app.route('/patientlist')
def patientlist():
    return render_template('patient.html')





@app.route('/add_user', methods=['POST'])
def add_user():
    patientid = request.form['patientid']
    name = request.form['name'],
    con = request.form['con'],
    pressure = request.form['pressure'],
    oxygen = request.form['oxygen'],
    pulse = request.form['pulse'],
    age = request.form['age'],
    dob = request.form['dob'],
    caretaker = request.form['caretaker'],
    place = request.form['place']
    for data in datas:
        if data['patientid'] == patientid:
            return redirect(url_for('user_add'))

    data = {
        'patientid': patientid,
        'name': name,
        'con': con,
        'pressure': pressure,
        'oxygen': oxygen,
        'pulse': pulse,
        'age': age,
        'dob': dob,
        'caretaker': caretaker,
        'place': place,
    }
    datas.append(data)
    save_data()

    return redirect(url_for('user_add'))


@app.route('/delete_user', methods=['POST'])
def delete_user():
    patientid = request.form['patientid']

    for data in datas:
        if data['patientid'] == patientid:
            datas.remove(data)
            save_data()
            return redirect(url_for('user_add'))

    return redirect(url_for('patientlist'))


@app.route('/patient/<patientid>')
def patient_details(patientid):
    for data in datas:
        if data['patientid'] == patientid:
            return render_template('patient_details.html', patient=data)
    return 'Patient not found'






















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
    

