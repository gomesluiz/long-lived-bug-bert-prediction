import re
import pandas as pd
import numpy as np

def clean_data_fn(text):
    """Clean and convert a text to store only alphabetical characters 
       in lower case.
       
    Args:
        text (str): a text string.
    
    Returns:
        text (str): a text string converted.
    """
    text = text.lower()
    text = re.sub(r"([?.!,¿])", r" ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text=re.sub(r'@\w+', '',text)
    return text

def load_data_fn(file_path):
    """Read and clean a bug report data set.

    Args:
        filepath (str): a complete filename path.

    Returns:
        result (dataframe): a bug report dataframe.

    """
    reports = pd.read_csv(file_path, encoding='utf8', sep=',', parse_dates=True
      ,low_memory=False)

    reports.dropna(inplace=True)
    reports['long_description'] = reports['long_description'].map(clean_data_fn)
    reports['long_description'] = reports['long_description'].replace('', np.nan)
    
    result = reports.loc[:, ('long_description', 'severity_category')]
    result.dropna(inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result

def convert_to_ordinal_fn(severity):
    """Convert severity category to ordinal.

    Args:
        severity (str): a severity category.

    Returns:
        severity (int): ordinal value of severity.
    """
    categories={'trivial': 0,
                'minor': 1,
                'major': 2,
                'critical': 3,
                'blocker': 4}
    return categories.get(severity, 'Invalid severity category!')

def convert_to_categorical_fn(code):
    """Convert severity ordinal to category.

    Args:
        code (int): a severity ordinal.

    Returns:
        severity (string): categorical value of severity.
    """
    ordinals={0:'trivial',
              1:'minor',
              2:'major',
              3:'critical',
              4:'blocker'}
    return ordinals.get(code, 'Invalid severity ordinal!')