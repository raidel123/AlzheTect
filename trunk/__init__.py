from flask import Flask, render_template

import glob, os, sys
import pandas as pd

app = Flask(__name__)
appContext = os.path.abspath(os.path.dirname(__file__))
'''
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
'''

sys.path.append(appContext + "/src/utils")

import dbconnect as db

@app.route('/', methods=['GET', 'POST'])
@app.route('/home/', methods=['GET', 'POST'])
def HomePage():
    title="Alzhetect"

    return render_template("index.html", title=title)

@app.route('/stats/', methods=['GET', 'POST'])
def StatsPage():
    # connect to DB
    checkboxes = [{"value":"PTGENDER", "label":"Gender"},
                  {"value":"AGE", "label":"Age"},
                  {"value":"PTRACCAT", "label":"Race"},
                  {"value":"PTETHCAT", "label":"Ethnicity"},
                  {"value":"PTEDUCAT", "label":"Education"},
                  {"value":"PTMARRY", "label":"Marital Status"}]

    return render_template("stats.html", checkboxes=checkboxes)

@app.route('/alzhetect/', methods=['GET', 'POST'])
def AlzhetectPage():
    return render_template("alzhetect.html")

@app.route('/contact/', methods=['GET', 'POST'])
def ContactPage():
    return render_template("contact.html")

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run()
