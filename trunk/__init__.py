from flask import Flask, render_template

import glob, os, sys
import pandas as pd

app = Flask(__name__)
appContext = os.path.abspath(os.path.dirname(__file__))

# get main project path (in case this file is compiled alone)
if os.name == 'nt':
    # Windows
    rootContext = os.getcwd().split('\\')
else:
    # Ubuntu
    rootContext = os.getcwd().split('/')

rootContext = '/'.join(rootContext[:rootContext.index('AlzheTect') + 1])

sys.path.append(rootContext + "/trunk/src/utils")

import dbconnect as db

@app.route('/', methods=['GET', 'POST'])
@app.route('/home/', methods=['GET', 'POST'])
def HomePage():
    title="Alzhetect"

    return render_template("index.html", title=title)

@app.route('/stats/', methods=['GET', 'POST'])
def StatsPage():
    # connect to DB
    conn = db.OpenConnection()

    checkboxes = [{"value":"PTGENDER", "label":"Gender"},
                  {"value":"AGE", "label":"Age"},
                  {"value":"PTRACCAT", "label":"Race"},
                  {"value":"PTETHCAT", "label":"Ethnicity"},
                  {"value":"PTEDUCAT", "label":"Education"},
                  {"value":"PTMARRY", "label":"Marital Status"}
                  ]

    genderStats = {"Male":db.QueryDB("SELECT COUNT(PTGENDER) FROM patients WHERE PTGENDER='Male';", conn).iloc[0]['COUNT(PTGENDER)'],
                   "Female":db.QueryDB("SELECT COUNT(PTGENDER) FROM patients WHERE PTGENDER='Female';", conn).iloc[0]['COUNT(PTGENDER)']
                  }

    ageStats = {"X < 60":db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE<60;", conn).iloc[0]['COUNT(AGE)'],
                "60 <= X < 70":db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=60 AND AGE<70;", conn).iloc[0]['COUNT(AGE)'],
                "70 <= X < 80":db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=70 AND AGE<80;", conn).iloc[0]['COUNT(AGE)'],
                "80 <= X < 90":db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=80 AND AGE<90;", conn).iloc[0]['COUNT(AGE)'],
                "X >=  90":db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=90;", conn).iloc[0]['COUNT(AGE)']
                }

    print genderStats
    print ageStats

    db.CloseConnection(conn)

    return render_template("stats.html", checkboxes=checkboxes, genderStats=genderStats)

@app.route('/alzhetect/', methods=['GET', 'POST'])
def AlzhetectPage():
    return render_template("alzhetect.html")

@app.route('/contact/', methods=['GET', 'POST'])
def ContactPage():
    return render_template("contact.html")

# ----------------------------------------------------------------------------

def GenderStats(data):
    print data

if __name__ == "__main__":
    app.run()
