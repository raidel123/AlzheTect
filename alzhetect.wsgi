#! /usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/Alzhetect/")

from trunk import app as application
application.secret_key = 'freekeyforthesite'
