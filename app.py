# Standard packages.
import io
import os
import sys
sys.path.append('1-data-preparation')
import xml.etree.ElementTree as ET

# Third-party packages.
import numpy as np
import requests
import torch
import transformers as ppb
import xgboost as xgb

from flask import Flask, flash, render_template, request, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

# Local packages.
from data_preparation import clean_data_fn, convert_to_categorical_fn

# Setup application.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'wP4xQ8hU1jJ5oI1c'
bootstrap = Bootstrap(app)

# Custom error messages.
BUG_NOT_FOUND = "Bug report not found!"
BUG_DESCRIPTION_NOT_FOUND = "Bug report description not found!"

# Define custom form.
class InputForm(FlaskForm):
    bug_id = StringField(validators=[DataRequired()])


# Create DistilBERT model and tokenizer 
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, 
    ppb.DistilBertTokenizer, 
    'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model     = model_class.from_pretrained(pretrained_weights)

@app.route('/', methods=['GET', 'POST'])
def index():
    form        = InputForm(request.form)
    severity    = 'undefined'
    description = ''
    if form.validate_on_submit():
        description = get_description(form.bug_id.data) 
        if description not in [BUG_NOT_FOUND, BUG_DESCRIPTION_NOT_FOUND]:
            X = extract_features(clean_data_fn(description))
            severity = convert_to_categorical_fn(make_prediction(X)) 
        else:
            severity    = 'undefined'
            flash(description)
            description = ''

    return render_template('index.html', form=form, description=description, severity=str(severity))

def get_description(id):
    """Gets a bug report description from Mozilla Bugzilla project
       associated to the id passed as parameter.
    
    Args:
        id (id): the bug id number.
    
    Returns:
        description (str): the bug description.
    """

    url = f"https://bugzilla.mozilla.org/show_bug.cgi?ctype=xml&id={id}"

    response = requests.get(url)
    if "InvalidBugId" in response.text:
        return BUG_NOT_FOUND 

    file = io.StringIO(response.text)
    tree = ET.parse(file)
    root = tree.getroot()
    description = root.findall('./bug/long_desc/thetext')
    if description is None:
        return BUG_DESCRIPTION_NOT_FOUND

    return description[0].text

def extract_features(description, max_len=128):
    """Extract features from description using DistilBERT.

    Args:
        description(str) : a bug report description.
        max_len (int) : maximum number of tokens to consider.

    Returns:
        features (array): a vector of features.
    """

    sentence  = ' '.join(description.split()[:max_len])
    tokens    = tokenizer.encode(sentence, add_special_tokens=True) 
    padded    = np.array([tokens + [0]*(max_len-len(tokens))])
    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:,0,:].numpy()

    return features
    
def make_prediction(X):
    filename = os.path.join('data', 'model', 'final-model.bin')
    #dt = xgb.DMatrix(X)
    clf = xgb.XGBClassifier()
    bst = xgb.Booster({'nthread':4})
    bst.load_model(filename)
    clf._Booster = bst
    return clf.predict(X)[0]


