from flask import Flask,render_template,url_for,request,redirect, make_response
import random
import json
from time import time
from random import random
from flask import Flask, render_template, make_response
from datetime import datetime
import pandas as pd
import numpy as np
app = Flask(__name__)
itercount = 0
t = np.arange(0,1440,8)




@app.route('/', methods=["GET", "POST"])
def main():
    return render_template('plotlivedata1.html')

@app.route('/BGL', methods =["GET", "POST"])
def BGL(itercount=0):
    G,I=dummyGValues() # should be floats
    data = [time()*1000, G, I]
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    itercount += 1
    return data


if __name__ == "__main__":
    app.run(debug=True)

# Df = pd.read_csv('simdata.csv')
# Df = Dt.copy()
# Glu = Df.iloc[:, 0]
# F = Df.iloc[:, 2]
# I = Df.iloc[:, 1]
# Gg = []
# Ig = []
# Wp = []
# e=[]
# y1 = []

# def initialize(Glu):
#     n = len(Glu)
#     lam = 0.98
#     delta = 1e2
#     P = delta * np.identity(3)
#     w1 = np.matrix(np.zeros(3))
#     N = 4# set the prediction horizon
#     return n,lam,delta,P,W1,N
#
# hl = 180
# ul = 100
# def GmaxGmin():
#     Gmax = hl * np.ones((N, 1))
#     Gmin = ul * np.ones((N, 1))
#     return Gmax,Gmin

# def manage(Glu,F,I,Gg,Ig):
#     n,lam,delta,P,W1,N= initialize(Glu)
#     w = w1.T
#     # initial condition
#     Gk = Glu[0]
#     Gmax,Gmin = GmaxGmin()
#     # Constraints
#     for i in range(1, n-1):
#             u1 = np.matrix([I[i], F[i], Glu[i]])
#             u12 = np.array(u1)
#             u = u1.T        #,Glu[i+2]]
#             phi = np.matrix(u.T*P)
#             k = phi.T/(lam+phi*u)
#             y = w.T*u
#             yp = pd.DataFrame(y)
#             y1.append(yp[0])
#             e = Glu[i+1]-y
#             w = w+k*e
#             P = (P-k*phi)/lam
#             b = w[0]
#             c1 = w[1]
#             a = w[2]
#             Wp.append(w)
#             Cal_A = np.zeros((N, 1))
#             for ct in range(0, N, 1):
#                 Cal_A[ct] = a ** (ct + 1)
#             Cal_B = np.zeros((N, N))
#             Cal_B[0, 0] = b
#             for k in range(1, N, 1):
#                 Cal_B[k, :] = a * Cal_B[k - 1, :]
#                 Cal_B[k, k] = b
#             Cal_B = np.asmatrix(Cal_B)
#             Cal_C = np.zeros((N, N))
#             Cal_C[0, 0] = c1
#             for kt in range(1, N, 1):
#                 Cal_C[kt, :] = a * Cal_C[kt - 1, :]
#                 Cal_C[kt, kt] = c1
#             Cal_C = np.asmatrix(Cal_C)
#         f = np.vstack([np.vstack([np.ones((N, 1))]),
#                    np.vstack([1000.0 * np.ones((N, 1))]),
#                    ])
#         Gnew = np.vstack([np.hstack([Cal_B, np.eye((N))]),
#                       np.hstack([-Cal_B, -np.eye((N))]),
#                       np.hstack([np.eye((N)), np.zeros((N, N))]),
#                       np.hstack([-np.eye((N)), np.zeros((N, N))])
#                       ])
#         FF1 = np.vstack([Gmax - (np.matrix(Cal_A) * Gk) - (np.matrix(Cal_C) * (u12[0, 1] * np.ones((N, 1))))])
#         FF2 = np.vstack([-Gmin + (np.matrix(Cal_A) * Gk) + (np.matrix(Cal_C) * (u12[0, 1] * np.ones((N, 1))))])
#         FF3 = np.vstack([10 * np.ones((N, 1))])
#         FF4 = np.vstack([-0 * np.ones((N, 1))])
#         FF = np.vstack([FF1, FF2, FF3, FF4])
#
#         # solve the LP
#         lp = glpk.LPX()
#         lp.name = 'BGL control'
#         lp.obj.maximize = False
#         lp.rows.add(16)  #
#
#         ### bounds
#         lp.rows[0].bounds = None, float(FF[0, :])
#         lp.rows[1].bounds = None, float(FF[1, :])
#         lp.rows[2].bounds = None, float(FF[2, :])
#         lp.rows[3].bounds = None, float(FF[3, :])
#         lp.rows[4].bounds = None, float(FF[4, :])
#         lp.rows[5].bounds = None, float(FF[5, :])
#         lp.rows[6].bounds = None, float(FF[6, :])
#         lp.rows[7].bounds = None, float(FF[7, :])
#         lp.rows[8].bounds = None, float(FF[8, :])
#         lp.rows[9].bounds = None, float(FF[9, :])
#         lp.rows[10].bounds = None, float(FF[10, :])
#         lp.rows[11].bounds = None, float(FF[11, :])
#         lp.rows[12].bounds = None, float(FF[12, :])
#         lp.rows[13].bounds = None, float(FF[13, :])
#         lp.rows[14].bounds = None, float(FF[14, :])
#         lp.rows[15].bounds = None, float(FF[15, :])
#
#         # define columns
#         lp.cols.add(8)
#         lp.cols[0].name = "u0"
#         lp.cols[1].name = "u1"
#         lp.cols[2].name = "u2"
#         lp.cols[3].name = "u3"
#         lp.cols[4].name = "s0"
#         lp.cols[5].name = "s1"
#         lp.cols[6].name = "s2"
#         lp.cols[7].name = "s3"
#
#         ## Set column bounds
#         lp.cols[0].bounds = 0.0, 10.0
#         lp.cols[1].bounds = 0.0, 10.0
#         lp.cols[2].bounds = 0.0, 10.0
#         lp.cols[3].bounds = 0.0, 10.0
#         lp.cols[4].bounds = 0.0, 10.0
#         lp.cols[5].bounds = 0.0, 10.0
#         lp.cols[6].bounds = 0.0, 10.0
#         lp.cols[7].bounds = 0.0, 10.0
#
#         ##Set Objective coefficient
#         lp.obj[0] = 1
#         lp.obj[1] = 1
#         lp.obj[2] = 1
#         lp.obj[3] = 1
#         lp.obj[4] = 100
#         lp.obj[5] = 100
#         lp.obj[6] = 100
#         lp.obj[7] = 100
#
#         # for c in lp.cols:
#         # c.bounds = 0.0, 0.0
#         # lp.obj[:] = [1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0]
#
#
#         lp.matrix = [float(Gnew[0, 0]),
#                      float(Gnew[0, 1]),
#                      float(Gnew[0, 2]),
#                      float(Gnew[0, 3]),
#                      float(Gnew[0, 4]),
#                      float(Gnew[0, 5]),
#                      float(Gnew[0, 6]),
#                      float(Gnew[0, 7]),
#                      float(Gnew[1, 0]),
#                      float(Gnew[1, 1]),
#                      float(Gnew[1, 2]),
#                      float(Gnew[1, 3]),
#                      float(Gnew[1, 4]),
#                      float(Gnew[1, 5]),
#                      float(Gnew[1, 6]),
#                      float(Gnew[1, 7]),
#                      float(Gnew[2, 0]),
#                      float(Gnew[2, 1]),
#                      float(Gnew[2, 2]),
#                      float(Gnew[2, 3]),
#                      float(Gnew[2, 4]),
#                      float(Gnew[2, 5]),
#                      float(Gnew[2, 6]),
#                      float(Gnew[2, 7]),
#                      float(Gnew[3, 0]),
#                      float(Gnew[3, 1]),
#                      float(Gnew[3, 2]),
#                      float(Gnew[3, 3]),
#                      float(Gnew[3, 4]),
#                      float(Gnew[3, 5]),
#                      float(Gnew[3, 6]),
#                      float(Gnew[3, 7]),
#                      float(Gnew[4, 0]),
#                      float(Gnew[4, 1]),
#                      float(Gnew[4, 2]),
#                      float(Gnew[4, 3]),
#                      float(Gnew[4, 4]),
#                      float(Gnew[4, 5]),
#                      float(Gnew[4, 6]),
#                      float(Gnew[4, 7]),
#                      float(Gnew[5, 0]),
#                      float(Gnew[5, 1]),
#                      float(Gnew[5, 2]),
#                      float(Gnew[5, 3]),
#                      float(Gnew[5, 4]),
#                      float(Gnew[5, 5]),
#                      float(Gnew[5, 6]),
#                      float(Gnew[5, 7]),
#                      float(Gnew[6, 0]),
#                      float(Gnew[6, 1]),
#                      float(Gnew[6, 2]),
#                      float(Gnew[6, 3]),
#                      float(Gnew[6, 4]),
#                      float(Gnew[6, 5]),
#                      float(Gnew[6, 6]),
#                      float(Gnew[6, 7]),
#                      float(Gnew[7, 0]),
#                      float(Gnew[7, 1]),
#                      float(Gnew[7, 2]),
#                      float(Gnew[7, 3]),
#                      float(Gnew[7, 4]),
#                      float(Gnew[7, 5]),
#                      float(Gnew[7, 6]),
#                      float(Gnew[7, 7]),
#                      float(Gnew[8, 0]),
#                      float(Gnew[8, 1]),
#                      float(Gnew[8, 2]),
#                      float(Gnew[8, 3]),
#                      float(Gnew[8, 4]),
#                      float(Gnew[8, 5]),
#                      float(Gnew[8, 6]),
#                      float(Gnew[8, 7]),
#                      float(Gnew[9, 0]),
#                      float(Gnew[9, 1]),
#                      float(Gnew[9, 2]),
#                      float(Gnew[9, 3]),
#                      float(Gnew[9, 4]),
#                      float(Gnew[9, 5]),
#                      float(Gnew[9, 6]),
#                      float(Gnew[9, 7]),
#                      float(Gnew[10, 0]),
#                      float(Gnew[10, 1]),
#                      float(Gnew[10, 2]),
#                      float(Gnew[10, 3]),
#                      float(Gnew[10, 4]),
#                      float(Gnew[10, 5]),
#                      float(Gnew[10, 6]),
#                      float(Gnew[10, 7]),
#                      float(Gnew[11, 0]),
#                      float(Gnew[11, 1]),
#                      float(Gnew[11, 2]),
#                      float(Gnew[11, 3]),
#                      float(Gnew[11, 4]),
#                      float(Gnew[11, 5]),
#                      float(Gnew[11, 6]),
#                      float(Gnew[11, 7]),
#                      float(Gnew[11, 0]),
#                      float(Gnew[12, 1]),
#                      float(Gnew[12, 2]),
#                      float(Gnew[12, 3]),
#                      float(Gnew[12, 4]),
#                      float(Gnew[12, 5]),
#                      float(Gnew[12, 6]),
#                      float(Gnew[12, 7]),
#                      float(Gnew[13, 0]),
#                      float(Gnew[13, 1]),
#                      float(Gnew[13, 2]),
#                      float(Gnew[13, 3]),
#                      float(Gnew[13, 4]),
#                      float(Gnew[13, 5]),
#                      float(Gnew[13, 6]),
#                      float(Gnew[13, 7]),
#                      float(Gnew[14, 0]),
#                      float(Gnew[14, 1]),
#                      float(Gnew[14, 2]),
#                      float(Gnew[14, 3]),
#                      float(Gnew[14, 4]),
#                      float(Gnew[14, 5]),
#                      float(Gnew[14, 6]),
#                      float(Gnew[14, 7]),
#                      float(Gnew[15, 0]),
#                      float(Gnew[15, 1]),
#                      float(Gnew[15, 2]),
#                      float(Gnew[15, 3]),
#                      float(Gnew[15, 4]),
#                      float(Gnew[15, 5]),
#                      float(Gnew[15, 6]),
#                      float(Gnew[15, 7])]
#         lp.simplex()
#         J = lp.obj.value
#         print('The value of J', J)
#         # u = float(J)
#         u0 = lp.cols[1].primal
#         print('The control input is', u0)
#         Gk = a * Gk + b * u0 + c1 * u[1];
#         Gg.append(float(Gk[0]))
#         Ig.append(u0)
#
#
#         return t[i],Gk[0],u0











#@app.route('/data', methods=["GET", "POST"])
# def data():
#     # Data Format
#     # [TIME, Temperature, Humidity]
#     BGL = random() * 100
#     Insulin = random() * 55
#     now = datetime.now()
#     current_time = now.strftime('%H:%M:%S')
#     data = [time()*1000, BGL, Insulin]
#     response = make_response(json.dumps(data))
#     response.content_type = 'application/json'
#     return response
#



