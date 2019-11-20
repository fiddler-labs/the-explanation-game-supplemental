import io
import pathlib
import zipfile
from typing import Tuple

import pandas as pd
import requests
import yaml


def download_uci_adult(data_dir: pathlib.Path) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads the UCI Adult Income dataset and caches it as pickled
    pandas dataframes"""
    train_pkl = data_dir / 'train_df.pkl'
    test_pkl = data_dir / 'test_df.pkl'

    if data_dir.exists() and len(tuple(data_dir.iterdir())) > 1:
        train_df = pd.read_pickle(train_pkl)
        test_df = pd.read_pickle(test_pkl)
    else:
        # download the dataset from UCI and save to a gitignored directory
        data_dir.mkdir(exist_ok=True)
        with (data_dir / '.gitignore').open('w') as gitignore:
            gitignore.write('*.pkl\n')
        columns = [
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capitalgain',
            'capitalloss',
            'hoursperweek',
            'native-country',
            'class'
        ]
        train_url = 'https://archive.ics.uci.edu/ml/' \
                    'machine-learning-databases/adult/adult.data'
        test_url = 'https://archive.ics.uci.edu/ml/' \
                   'machine-learning-databases/adult/adult.test'
        train_df = pd.read_csv(train_url, header=None, skipinitialspace=True,
                               names=columns)
        test_df = pd.read_csv(test_url, header=None, skiprows=1,
                              skipinitialspace=True, names=columns)
        train_df.to_pickle(train_pkl)
        test_df.to_pickle(test_pkl)

    # remove extra periods in test labels
    test_df['class'] = test_df['class'].str.replace('.', '')

    # handle categorical datatypes by first concatenating,
    # then type converting, then splitting
    train_len = train_df.shape[0]
    full_df = train_df.append(test_df)
    del train_df, test_df
    for column in full_df.select_dtypes('object'):
        full_df[column] = full_df[column].astype('category')
    train_df = full_df.iloc[:train_len].copy()
    test_df = full_df.iloc[train_len:].copy()
    del full_df

    return train_df, test_df


def download_uci_bikeshare(data_dir: pathlib.Path) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads the UCI Bikeshare dataset and caches it as pickled
    pandas dataframes"""

    # Downlaod the dataset (with caching to disk)
    train_pkl = data_dir / 'train_df.pkl'
    test_pkl = data_dir / 'test_df.pkl'

    if data_dir.exists() and len(tuple(data_dir.iterdir())) > 1:
        train_df = pd.read_pickle(train_pkl)
        test_df = pd.read_pickle(test_pkl)
    else:
        # import and save the dataset to a gitignored directory
        data_dir.mkdir(exist_ok=True)
        with (data_dir / '.gitignore').open('w') as gitignore:
            gitignore.write('*.pkl\n')

        # download the dataset from UCI
        zip_url = 'https://archive.ics.uci.edu/ml/' \
                  'machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
        z = zipfile.ZipFile(io.BytesIO(requests.get(zip_url).content))
        dtypes = {
            'holiday': 'bool',
            'workingday': 'bool',
            'weathersit': 'category',
            'season': 'category'
        }
        with z.open('hour.csv') as csv:
            full_df = pd.read_csv(csv, dtype=dtypes)

        # split train/test by year
        is_2011 = full_df['yr'] == 0
        train_df = full_df[is_2011].reset_index(drop=True)
        test_df = full_df[~is_2011].reset_index(drop=True)

        # serialize datasets to disk
        train_df.to_pickle(train_pkl)
        test_df.to_pickle(test_pkl)

    return train_df, test_df


def download_loans(data_dir: pathlib.Path) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads a peer-to-peer lending dataset and caches it as pickled
    pandas dataframes"""

    # Downlaod the dataset (with caching to disk)
    train_pkl = data_dir / 'train_df.pkl'
    test_pkl = data_dir / 'test_df.pkl'

    if data_dir.exists() and len(tuple(data_dir.iterdir())) > 1:
        train_df = pd.read_pickle(train_pkl)
        test_df = pd.read_pickle(test_pkl)
    else:
        # import and save the dataset to a gitignored directory
        data_dir.mkdir(exist_ok=True)
        with (data_dir / '.gitignore').open('w') as gitignore:
            gitignore.write('*.pkl\n')

        # define URLs
        base_url = 'https://raw.githubusercontent.com/fiddler-labs/' \
                   'p2p-lending-data/master/p2p_loans_470k/'
        feature_schema_url = base_url + 'feature_schema.yaml'
        label_schema_url = base_url + 'label_schema.yaml'
        train_feature_url = base_url + 'train/train_features.csv.gz'
        train_label_url = base_url + 'train/train_labels.csv.gz'
        test_feature_url = base_url + 'test/test_features.csv.gz'
        test_label_url = base_url + 'test/test_labels.csv.gz'

        # load the schemas
        feature_schema = yaml.load(io.BytesIO(
            requests.get(feature_schema_url).content))
        label_schema = yaml.load(io.BytesIO(
            requests.get(label_schema_url).content))

        # import the data from remote using the schema
        train_df = pd.read_csv(train_feature_url, **feature_schema)
        train_labels = pd.read_csv(train_label_url, **label_schema)

        # re-use the train datatypes
        # match the schema
        test_schema = feature_schema.copy()
        test_schema['dtype'] = {**train_df.dtypes.to_dict(),
                                'earliest_cr_line': 'object'}

        test_df = pd.read_csv(test_feature_url, **test_schema)
        test_labels = pd.read_csv(test_label_url, **label_schema)

        # drop labels that aren't useful out-of-the-box
        # i.e. those that are strings or dates
        drop_features = train_df.select_dtypes(
            ['datetime', 'O']).columns.tolist()
        train_df.drop(columns=drop_features, inplace=True)
        test_df.drop(columns=drop_features, inplace=True)

        # set target
        train_df['charged_off'] = train_labels['loan_status'].eq('Charged Off')
        test_df['charged_off'] = test_labels['loan_status'].eq('Charged Off')

        # ensure dtypes line up exactly
        assert train_df.dtypes.eq(test_df.dtypes).all()
        assert train_labels.dtypes.eq(test_labels.dtypes).all()

        # serialize datasets to disk
        train_df.to_pickle(train_pkl)
        test_df.to_pickle(test_pkl)

    return train_df, test_df
