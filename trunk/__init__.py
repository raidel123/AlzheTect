from flask import Flask, render_template, request
from werkzeug import secure_filename

import glob, os, sys
import pandas as pd
import numpy as np
import math
import io

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
import mlearning as ml

@app.route('/', methods=['GET', 'POST'])
@app.route('/home/', methods=['GET', 'POST'])
def HomePage():
    title="Alzhetect"

    conn = db.OpenConnection(appContext + '/src/sqldb/alzhetect.db')

    query_features = db.QueryDB("SELECT * FROM selected_features;", conn)
    selected_features = {"th" : query_features.keys(),
                         "td" : query_features.values.tolist()}

    # print "keys", query_features.keys()
    # print "values", query_features.values.tolist()

    db.CloseConnection(conn)

    return render_template("index.html", title=title, selected_features=selected_features)

@app.route('/stats/', methods=['GET', 'POST'])
def StatsPage():

    conn = db.OpenConnection(appContext + '/src/sqldb/alzhetect.db')

    query_features = db.QueryDB("SELECT * FROM selected_features;", conn)

    checkboxes = GenerateCheckbox(conn, query_features)

    selected_features = {"th" : query_features.keys(),
                         "td" : query_features.values.tolist()}

    db.CloseConnection(conn)
    return render_template("stats.html", checkboxes=checkboxes, selected_features=selected_features)

@app.route('/alzhetect/', methods=['GET', 'POST'])
def AlzhetectPage():

    conn = db.OpenConnection(appContext + '/src/sqldb/alzhetect.db')

    # query_features =

    # checkboxes = GenerateCheckbox(conn, query_features)

    #selected_features = {"th" : query_features.keys(), "td" : query_features.values.tolist()}
    field_name = db.QueryDB("SELECT Field_Name FROM selected_features;", conn).values.tolist()
    field_type = db.QueryDB("SELECT Measurement_Type FROM selected_features;", conn).values.tolist()

    patient_input_fields = {"field_name":field_name, "field_type":field_type}

    '''
    patient_input_fields =  [
                'MMSE_bl',
                'CDRSB',
                'ADAS13',
                'ADAS11',
                'RAVLT_immediate',
                'MMSE',
                'APOE4',
                'LEFT_AMYGDALA_UCBERKELEYAV45_10_17_16',
                'ST88SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                'ST29SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                'Hippocampus',
                'ST82TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                'ST39TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                'ST82TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                'ST83CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                'ST30SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                'ST109TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                'AV45',
                'WholeBrain',
                'LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16',
                ]
    '''

    results = [0]

    db.CloseConnection(conn)

    return render_template("alzhetect.html", patient_input_fields=patient_input_fields, results=results)

@app.route('/contact/', methods=['GET', 'POST'])
def ContactPage():
    return render_template("contact.html")

@app.route('/uploader/', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      in_file = f.read()
      # print type(in_file)
      # print pd.DataFrame([sub.split(",")]f.read())
      # f.save("results/uploads/upload.csv") # + secure_filename(f.filename))
      # redict_csv = ml.GetModelDataCSV(io.StringIO(unicode(in_file)))
      # print predict_csv
      # return model_dp
      # print io.StringIO(unicode(in_file)).getvalue()

      results = ml.knn_predict(model_loc=appContext+"/src/trained_model/knn/knnmodel2.pickle", input_data=io.StringIO(unicode(in_file)), output_file=appContext+"/static/results.csv")

      return render_template("contact.html", results=results.values.tolist())

@app.route('/fieldloader/', methods = ['GET', 'POST'])
def upload_fields():
    if request.method == 'POST':
        print request.form.to_dict()
    return "Field Loader Success"

# ----------------------------------------------------------------------------
def GenerateCheckbox(conn, query_features):
    append_table = StaticCheckboxes(conn)
    for index, row in query_features.iterrows():
        field = row['Field_Name']
        field_data_stats = SimpleQuery(field, conn)
        # field_data_stats = field_data_stats[~np.isnan(field_data_stats)]
        final_data = []
        unknown = 0
        # print field_data_stats.describe().to_dict()

        for nrow in field_data_stats.values.tolist():
            try:
                data_val = float(nrow[0])
            except (ValueError, TypeError) as e:
                data_val = None

            if data_val == ' ' or data_val is None or math.isnan(data_val):
                unknown += 1
            else:
                final_data.append(data_val)

        if len(final_data) == 0:
            continue

        final_data.sort()
        fd_mean = np.mean(final_data)
        fd_std = np.std(final_data)

        stats = []
        # chunk = len(final_data)/4
        bottom_indx = 0
        top_ind = LastLessThanIndex(int(fd_mean - fd_std), final_data)

        slice1 = final_data[bottom_indx:top_ind+1]
        stats.append(["X < " + str(slice1[-1]), len(slice1)-1])

        bottom_indx = top_ind
        top_ind = LastLessThanIndex(int(fd_mean), final_data)
        slice1 = final_data[bottom_indx:top_ind+1]
        stats.append([str(slice1[0]) + " <= X < " + str(slice1[-1]), len(slice1)-1])

        bottom_indx = top_ind
        top_ind = LastLessThanIndex(int(fd_mean + fd_std), final_data)
        slice1 = final_data[bottom_indx:top_ind+1]
        stats.append([str(slice1[0]) + " <= X < " + str(slice1[-1]), len(slice1)-1])

        bottom_indx = top_ind
        slice1 = final_data[bottom_indx:]
        stats.append(["X >= " + str(slice1[0]), len(slice1)])

        # stats.append(['Unknown', unknown])

        append_table.append({"value":row['Field_Name'], "label":row['Measurement_Type'], "stats":stats, "description":row['Description']})

    return append_table

def SimpleQuery(field, conn):
    return db.QueryDB("SELECT " + field + " FROM patients WHERE DX_bl='AD';",conn)

def LastLessThanIndex(less_val, ilist):
    for i in range(len(ilist)):
        if ilist[i] >= less_val:
            return i

def StaticCheckboxes(conn):
    genderStats = [["Male", db.QueryDB("SELECT COUNT(PTGENDER) FROM patients WHERE PTGENDER='Male' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTGENDER)']],
                 ["Female", db.QueryDB("SELECT COUNT(PTGENDER) FROM patients WHERE PTGENDER='Female' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTGENDER)']]]

    ageStats = [["X < 60", db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE<60 AND DX_bl='AD';", conn).iloc[0]['COUNT(AGE)']],
              ["60 <= X < 70", db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=60 AND AGE<70 AND DX_bl='AD';", conn).iloc[0]['COUNT(AGE)']],
              ["70 <= X < 80", db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=70 AND AGE<80 AND DX_bl='AD';", conn).iloc[0]['COUNT(AGE)']],
              ["80 <= X < 90", db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=80 AND AGE<90 AND DX_bl='AD';", conn).iloc[0]['COUNT(AGE)']],
              ["X >=  90", db.QueryDB("SELECT COUNT(AGE) FROM patients WHERE AGE>=90 AND DX_bl='AD';", conn).iloc[0]['COUNT(AGE)']]]

    raceStats = [["White", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='White' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTRACCAT)']],
               ["Black", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='Black' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTRACCAT)']],
               ["Asian", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='Asian' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTRACCAT)']],
               ["Am Indian/Alaskan", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='Am Indian/Alaskan' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTRACCAT)']],
               ["More than one", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='More than one' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTRACCAT)']],
               ["Unknown", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='Unknown' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTRACCAT)']],
               ["Hawaiian/Other PI", db.QueryDB("SELECT COUNT(PTRACCAT) FROM patients WHERE PTRACCAT='Hawaiian/Other PI' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTRACCAT)']]]

    ethStats = [["Not Hisp/Latino", db.QueryDB("SELECT COUNT(PTETHCAT) FROM patients WHERE PTETHCAT='Not Hisp/Latino' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTETHCAT)']],
              ["Hisp/Latino", db.QueryDB("SELECT COUNT(PTETHCAT) FROM patients WHERE PTETHCAT='Hisp/Latino' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTETHCAT)']],
              ["Unknown", db.QueryDB("SELECT COUNT(PTETHCAT) FROM patients WHERE PTETHCAT='Unknown' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTETHCAT)']]]

    eduStats = [["X < 5", db.QueryDB("SELECT COUNT(PTEDUCAT) FROM patients WHERE PTEDUCAT<5 AND DX_bl='AD';", conn).iloc[0]['COUNT(PTEDUCAT)']],
              ["5 <= X < 10", db.QueryDB("SELECT COUNT(PTEDUCAT) FROM patients WHERE PTEDUCAT>=5 AND PTEDUCAT<10 AND DX_bl='AD';", conn).iloc[0]['COUNT(PTEDUCAT)']],
              ["10 <= X < 15", db.QueryDB("SELECT COUNT(PTEDUCAT) FROM patients WHERE PTEDUCAT>=10 AND PTEDUCAT<15 AND DX_bl='AD';", conn).iloc[0]['COUNT(PTEDUCAT)']],
              ["15 <= X < 20", db.QueryDB("SELECT COUNT(PTEDUCAT) FROM patients WHERE PTEDUCAT>=15 AND PTEDUCAT<20 AND DX_bl='AD';", conn).iloc[0]['COUNT(PTEDUCAT)']],
              ["X >= 20", db.QueryDB("SELECT COUNT(PTEDUCAT) FROM patients WHERE PTEDUCAT>=20 AND DX_bl='AD';", conn).iloc[0]['COUNT(PTEDUCAT)']]]

    marryStats = [["Married", db.QueryDB("SELECT COUNT(PTMARRY) FROM patients WHERE PTMARRY='Married' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTMARRY)']],
                ["Divorced", db.QueryDB("SELECT COUNT(PTMARRY) FROM patients WHERE PTMARRY='Divorced' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTMARRY)']],
                ["Widowed", db.QueryDB("SELECT COUNT(PTMARRY) FROM patients WHERE PTMARRY='Widowed' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTMARRY)']],
                ["Never married", db.QueryDB("SELECT COUNT(PTMARRY) FROM patients WHERE PTMARRY='Never married' AND DX_bl='AD';", conn).iloc[0]['COUNT(PTMARRY)']],
                ["Unknown", db.QueryDB("SELECT COUNT(PTMARRY) FROM patients WHERE PTMARRY='Unknown'AND DX_bl='AD';", conn).iloc[0]['COUNT(PTMARRY)']]]

    checkboxes = [{"value":"PTGENDER", "label":"Gender", "stats":genderStats, "description":0},
                  {"value":"AGE", "label":"Age", "stats":ageStats, "description":0},
                  {"value":"PTRACCAT", "label":"Race", "stats":raceStats, "description":0},
                  {"value":"PTETHCAT", "label":"Ethnicity", "stats":ethStats, "description":0},
                  {"value":"PTEDUCAT", "label":"Education", "stats":eduStats, "description":0},
                  {"value":"PTMARRY", "label":"Marital Status", "stats":marryStats, "description":0},
                  ]

    return checkboxes

if __name__ == "__main__":
    app.run()
