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

    genderStats = [["Male", db.QueryDB("SELECT COUNT(PTGENDER) FROM patients WHERE PTGENDER='Male';", conn).iloc[0]['COUNT(PTGENDER)']],
                   ["Female", db.QueryDB("SELECT COUNT(PTGENDER) FROM patients WHERE PTGENDER='Female';", conn).iloc[0]['COUNT(PTGENDER)']]]

    ageStats = [["X < 60", db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE<60;", conn).iloc[0]['COUNT(AGE)']],
                ["60 <= X < 70", db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=60 AND AGE<70;", conn).iloc[0]['COUNT(AGE)']],
                ["70 <= X < 80", db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=70 AND AGE<80;", conn).iloc[0]['COUNT(AGE)']],
                ["80 <= X < 90", db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=80 AND AGE<90;", conn).iloc[0]['COUNT(AGE)']],
                ["X >=  90", db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=90;", conn).iloc[0]['COUNT(AGE)']]]

    raceStats = [["White", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='White';", conn).iloc[0]['COUNT(PTRACCAT)']],
                 ["Black", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='Black';", conn).iloc[0]['COUNT(PTRACCAT)']],
                 ["Asian", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='Asian';", conn).iloc[0]['COUNT(PTRACCAT)']],
                 ["Am Indian/Alaskan", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='Am Indian/Alaskan';", conn).iloc[0]['COUNT(PTRACCAT)']],
                 ["More than one", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='More than one';", conn).iloc[0]['COUNT(PTRACCAT)']],
                 ["Unknown", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='Unknown';", conn).iloc[0]['COUNT(PTRACCAT)']],
                 ["Hawaiian/Other PI", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='Hawaiian/Other PI';", conn).iloc[0]['COUNT(PTRACCAT)']]]

    ethStats = [["Not Hisp/Latino", db.QueryDB("SELECT COUNT(PTETHCAT) FROM patients WHERE PTETHCAT='Not Hisp/Latino';", conn).iloc[0]['COUNT(PTETHCAT)']],
                ["Hisp/Latino", db.QueryDB("SELECT COUNT(PTETHCAT) FROM patients WHERE PTETHCAT='Hisp/Latino';", conn).iloc[0]['COUNT(PTETHCAT)']],
                ["Unknown", db.QueryDB("SELECT COUNT(PTETHCAT) FROM patients WHERE PTETHCAT='Unknown';", conn).iloc[0]['COUNT(PTETHCAT)']]]

    eduStats = [["X < 5", db.QueryDB("SELECT COUNT(PTEDUCAT) FROM patients WHERE PTEDUCAT<5;", conn).iloc[0]['COUNT(PTEDUCAT)']],
                ["5 <= X < 10", db.QueryDB("SELECT COUNT(PTEDUCAT) FROM patients WHERE PTEDUCAT>=5 AND PTEDUCAT<10;", conn).iloc[0]['COUNT(PTEDUCAT)']],
                ["10 <= X < 15", db.QueryDB("SELECT COUNT(PTEDUCAT) FROM patients WHERE PTEDUCAT>=10 AND PTEDUCAT<15;", conn).iloc[0]['COUNT(PTEDUCAT)']],
                ["15 <= X < 20", db.QueryDB("SELECT COUNT(PTEDUCAT) FROM patients WHERE PTEDUCAT>=15 AND PTEDUCAT<20;", conn).iloc[0]['COUNT(PTEDUCAT)']],
                ["X >= 20", db.QueryDB("SELECT COUNT(PTEDUCAT) FROM patients WHERE PTEDUCAT>=20;", conn).iloc[0]['COUNT(PTEDUCAT)']]]

    marryStats = [["Married", db.QueryDB("SELECT COUNT(PTMARRY) FROM patients WHERE PTMARRY='Married';", conn).iloc[0]['COUNT(PTMARRY)']],
                  ["Divorced", db.QueryDB("SELECT COUNT(PTMARRY) FROM patients WHERE PTMARRY='Divorced';", conn).iloc[0]['COUNT(PTMARRY)']],
                  ["Widowed", db.QueryDB("SELECT COUNT(PTMARRY) FROM patients WHERE PTMARRY='Widowed';", conn).iloc[0]['COUNT(PTMARRY)']],
                  ["Never married", db.QueryDB("SELECT COUNT(PTMARRY) FROM patients WHERE PTMARRY='Never married';", conn).iloc[0]['COUNT(PTMARRY)']],
                  ["Unknown", db.QueryDB("SELECT COUNT(PTMARRY) FROM patients WHERE PTMARRY='Unknown';", conn).iloc[0]['COUNT(PTMARRY)']]]

    checkboxes = [{"value":"PTGENDER", "label":"Gender", "stats":genderStats},
                  {"value":"AGE", "label":"Age", "stats":ageStats},
                  {"value":"PTRACCAT", "label":"Race", "stats":raceStats},
                  {"value":"PTETHCAT", "label":"Ethnicity", "stats":ethStats},
                  {"value":"PTEDUCAT", "label":"Education", "stats":eduStats},
                  {"value":"PTMARRY", "label":"Marital Status", "stats":marryStats}]

    db.CloseConnection(conn)

    return render_template("stats.html", checkboxes=checkboxes)

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
