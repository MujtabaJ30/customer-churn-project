import pandas as pd
import joblib
import streamlit as st
import shap

DATA_PATH = 'data/processed/customer_data_with_predictions.csv'
MASTER_DATA_PATH = 'data/processed/master_df.csv'
MODEL_PATH = 'models/rf_churn_model.pkl'
EXPLAINER_PATH = 'models/shap_explainer.pkl'
FEATURE_NAMES_PATH = 'models/feature_names.pkl'
ENCODING_CONFIG_PATH = 'models/encoding_config.pkl'

COLS_TO_DROP = [
    'Customer ID', 'Churn Category', 'Churn Reason', 'Churn Score',
    'CLTV', 'Zip Code', 'Latitude', 'Longitude', 'City', 'Country', 'State',
    'Churn Label', 'Customer Status', 'Lat Long'
]


@st.cache_resource
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_master_data():
    return pd.read_csv(MASTER_DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_shap_explainer():
    return joblib.load(EXPLAINER_PATH)


@st.cache_resource
def load_feature_names():
    return joblib.load(FEATURE_NAMES_PATH)


@st.cache_resource
def load_encoding_config():
    return joblib.load(ENCODING_CONFIG_PATH)


def encode_customer(customer_row, config, feature_names):
    row = customer_row.drop(index=COLS_TO_DROP, errors='ignore')

    binary_map = config['binary_map']
    binary_cols = [c for c in config['binary_cols'] if c in row.index]
    row[binary_cols] = row[binary_cols].replace(binary_map).astype(int)

    row = row.to_frame().T.reset_index(drop=True)

    for col, cats in config['onehot_categories'].items():
        val = row[col].iloc[0]
        for cat in cats:
            row[f'{col}_{cat}'] = 1 if val == cat else 0
        row = row.drop(columns=[col])

    if 'Churn Value' in row.columns:
        row = row.drop(columns=['Churn Value'])

    row = row[feature_names]
    return row
