# AlzheTect
Using a Deep Neural Network(DNN), Support Vector Classifier(SVC), and Random Forest Regressor to detect early Alzheimer's disease using Python.

## Research Artifacts
- [Paper](https://github.com/raidel123/AlzheTect/blob/master/docs/Graduation_Final_Report.pdf)
- [Presentation](https://github.com/raidel123/AlzheTect/blob/master/docs/Presentation_%20AlzheTect.Bio-Marker%20Analysis%20For%20Early%20Alzheimer%E2%80%99s%20Classification.pdf)

## Requirements
The installation was performed on a [ Ubuntu 16.04 ](https://www.ubuntu.com/download/desktop) terminal.

#### System Requirements
* [ Python 2.x ](https://www.python.org/downloads/)
* [ Pip ](https://pip.pypa.io)
* [ Virtualenv ](https://virtualenv.pypa.io)

```
    $ pip install virtualenv
```

####  Installation Requirements
Setting up the virtual environment and installing requirements:
```
    $ virtualenv venv
    $ source venv\bin\activate
    $ pip install -r requirements.txt
```

Deactivating the virtual environment
```
    $ deactivate
```

#### To install certain components individually (if needed):
Tensorflow
```
    $ pip install tensorflow
```
Pandas
```
    $ pip install pandas
```

## Run
```
    $ python trunk/__init__.py
```

## Results
After executing the main script using the command above ([Run](#Run)), the results will be saved to the results directory located in "trunk/results". The results are based on the patients from the test file, located in the directory "trunk/src/test/TADPOLE_test_MCI.csv". Although the test file contains the patient diagnosis, during prediction it is omitted and then compared.
