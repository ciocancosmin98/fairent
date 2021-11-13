from flask import Flask, request
import tensorflow as tf
from training import custom_loss
from preprocessor import preprocess, getBaseDf, reverseNormPrice, getFeatures, keys
import pandas as pd
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('../data/model.h5', custom_objects={'custom_loss': custom_loss})
# Check its architecture
model.summary()

base_df = getBaseDf()
base_df = preprocess(base_df)

def get_value_from_request(req_data, key_name):
    value = req_data.get(key_name, None)

    if not value is None:
        return value

    # default values, the price here is put just to make the preprocessing work
    # it is later discarded
    if key_name == 'price':
        return '100'
    elif key_name == 'useful_area':
        return '50'
    elif key_name == 'n_bathrooms':
        return '1'
    elif key_name == 'n_kitchens':
        return '1'
    elif key_name == 'room_config':
        return 'detached'

    return ' '

def preprocess_json(req_data):
    dictData = {key: [] for key in keys} 
    for key in keys:
        value = get_value_from_request(req_data, key)
        dictData[key].append(value)

    df = pd.DataFrame.from_dict(dictData)

    return df

@app.route('/predict', methods=['POST'])
def index():
    request_data = request.get_json()

    df = preprocess_json(request_data)
    df = preprocess(df)

    df = pd.concat([base_df, df])
    _, features = getFeatures(df)

    features = features[-1, :]
    k = features.shape[0]
    features = features.reshape((1, k))
    """
    features = pd.get_dummies(df)
    features = features.iloc[-1:]

    print(features)
 
    # drop dummy price value
    features = features.drop('price', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)
    """

    predictions = model(features).numpy()
    predictions = reverseNormPrice(predictions)
    print(predictions)

    min_price = predictions.min()
    max_price = predictions.max()

    return f'\
    {{\n\
        "min_price": "{min_price}"\n\
        "max_price": "{max_price}"\n\
    }}'

if __name__ == "__main__":
    app.run(port=5000, debug=True)