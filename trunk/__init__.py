from flask import Flask, render_template


import glob, os

app = Flask(__name__)
appContext = os.path.abspath(os.path.dirname(__file__))

@app.route('/', methods=['GET', 'POST'])
@app.route('/home/', methods=['GET', 'POST'])
def homepage():
    # return "Hi there, how ya doin?"
    return render_template("index.html", title="Alzhetect", paragraph="Hello this is the site!")

'''
@app.route('/portfolio-single/', methods=['GET', 'POST'])
def portfolioInfo():
    return render_template("portfolio-single.html")
'''

if __name__ == "__main__":
    app.run()
