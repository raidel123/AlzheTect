from flask import Flask, render_template

import glob, os

app = Flask(__name__)
appContext = os.path.abspath(os.path.dirname(__file__))

@app.route('/', methods=['GET', 'POST'])
@app.route('/home/', methods=['GET', 'POST'])
def HomePage():
    # return "Hi there, how ya doin?"
    return render_template("index.html", title="Alzhetect", paragraph="Hello this is the site!")

@app.route('/stats/', methods=['GET', 'POST'])
def StatsPage():
    return render_template("stats.html")

@app.route('/alzhetect/', methods=['GET', 'POST'])
def AlzhetectPage():
    return render_template("alzhetect.html")

@app.route('/contact/', methods=['GET', 'POST'])
def ContactPage():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run()
