from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    data = [
        (1, 20),
        (2,100),
        (3, 40),
        (4, 50),
        (5, 20),
        (6, 100),
        (7, 40),
        (8, 50),
        (9, 20),
        (10, 100),
        (11, 40),
        (12, 50),
    ]
    labels = [row[0] for row in data]
    values = [row[1] for row in data]
    return render_template("graph.html", labels= labels, values=values)
