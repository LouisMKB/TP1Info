import re
import pandas as pd
import numpy as np
from loguru import logger

def load_data(file_path: str):
    """
    Charge le dataset.
    Args:
    - path (str): Chemin vers un fichier csv
    Returns:
    - pd.DataFrame: Le dataset
    """
    return pd.read_csv(file_path)


def replace_nan(df: pd.DataFrame) -> pd.DataFrame :
    """Met des valeurs NaN à la place de '?'

    Args:
        df (pd.DataFrame): dataset

    Returns:
        pd.DataFrame: dataset avec des valeurs NaN
    """
    return df.replace('?', np.nan)

def lower_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Change le nom des colonnes en minuscules

    Args:
        df (pd.DataFrame): dataset

    Returns:
        pd.DataFrame: dataset avec des colonnes au nom minuscule
    """
    data.columns = data.columns.str.lower()
    return data


def retain_first_cabin(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait la premiere lettre de la valeur de cabin

    Args:
    - data (pd.DataFrame): Le dataset.

    Returns:
    - pd.DataFrame: dataset avec la premiere lettre de cabin.
    """
    def get_first_cabin(row):
        try:
            return row.split()[0]
        except:
            return np.nan
    data['cabin'] = data['cabin'].apply(get_first_cabin)
    return data


def add_titles(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les titres de courtoisies des noms des passagers (Mr, Mrs, Miss,...)

    Args:
    - data (pd.DataFrame): Le dataset

    Returns:
    - pd.DataFrame: dataset avec une nouvelles colonnes 'title'.
    """
    def get_title(passenger):
        if re.search('Mrs', passenger):
            return 'Mrs'
        elif re.search('Mr', passenger):
            return 'Mr'
        elif re.search('Miss', passenger):
            return 'Miss'
        elif re.search('Master', passenger):
            return 'Master'
        else:
            return 'Other'

    data['title'] = data['name'].apply(get_title)
    return data

def type_conversion(data: pd.DataFrame, target: str = 'survived') -> pd.DataFrame:
    """
    Change le type des variables 'fare' et 'age' et des variables catégoriques et numériques

    Args:
    - data (pd.DataFrame): Le dataset.
    - target (str): Le nom de la variable cible

    Returns:
    - pd.DataFrame: Le nouveau data modifié
    """
    data['fare'] = data['fare'].astype('float')
    data['age'] = data['age'].astype('float')

    vars_num = [c for c in data.columns if data[c].dtypes != 'O' and c != target]
    vars_cat = [c for c in data.columns if data[c].dtypes == 'O'] 
    logger.info('Nombre de variables numériques: {}'.format(len(vars_num)))
    logger.info('Nombre de variables categoriques : {}'.format(len(vars_cat)))  

    return vars_num, vars_cat

def fill_by_median(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs manquantes par la médiane

    Args:
    - data (pd.DataFrame): Le dataset.

    Returns:
    - pd.DataFrame: Le dataset avec les nouvelles valeurs.
    """
    for var in ['age', 'fare']:

        # add missing indicator
        data[var+'_NA'] = np.where(data[var].isnull(), 1, 0)
        data[var+'_NA'] = np.where(data[var].isnull(), 1, 0)

        # replace NaN by median
        median_val = data[var].median()

        data[var].fillna(median_val, inplace=True)
        data[var].fillna(median_val, inplace=True)

    return data[['age', 'fare']].isnull().sum()