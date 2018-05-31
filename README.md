# AlzheTect
Using a Deep Neural Network(DNN) to detect early Alzheimer's disease using Tensorflow.

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
    $ python trunk/src/alzhetect.py
```
