import json
from os import name
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

nameConversion = {
    'Confort': 'comfort',
    'Etaj': 'floor',
    'Nr. balcoane': 'n_balconies',
    'Nr. garaje': 'n_garages',
    'Nr. bucătării': 'n_kitchens',
    'Regim înălţime': 'floors_config',
    'Suprafaţă utilă': 'useful_area',
    'Tip imobil': 'building_type',
    'Suprafaţă utilă totală': 'total_useful_area',
    'Nr. camere': 'n_rooms',
    'Nr. locuri parcare': 'n_parking_spaces',
    'pret': 'price',
    'Structură rezistenţă': 'building_material',
    'Nr. băi': 'n_bathrooms',
    'An construcţie': 'year_construction',
    'Suprafaţă construită': 'built_area',
    'Compartimentare': 'room_config'
}

keys = sorted({nameConversion[t] for t in nameConversion})

categorical = {
    'comfort',
    'floor',
    'n_balconies',
    'n_garages',
    'n_kitchens',
    'floors_config',
    'building_type',
    'n_rooms',
    'n_parking_spaces',
    'building_material',
    'n_bathrooms',
    'year_construction',
    'room_config'
}

"""
keys = {
    'comfort',
    'floor',
    'n_balconies',
    'n_garages',
    'n_kitchens',
    'floors_config',
    'useful_area',
    'building_type',
    'total_useful_area',
    'n_rooms',
    'n_parking_spaces',
    'price',
    'building_material',
    'n_bathrooms',
    'year_construction',
    'built_area',
    'room_config'
}
"""

def getKeyCounts(data, keys, doPrint=True):
    keyCounts = {key: 0 for key in keys}

    for item in data:
        for key in item:
            keyCounts[key] += 1

    if doPrint:
        print(keyCounts, '\n\n')

    return keyCounts

def getBaseDf():
    with open('../data/dataset.json', 'r') as file:
        data = json.load(file)

    keys = set([key for item in data for key in item])
    _ = getKeyCounts(data, keys, True)

    dictData = {nameConversion[key]: [] for key in keys} 
    for item in data:
        for key in keys:
            value = item.get(key, None)
            dictData[nameConversion[key]].append(value)

    df = pd.DataFrame.from_dict(dictData)

    return df

def reverseNormPrice(normPrice: np.ndarray, logged=False):
    maxPrice = 2000
    minPrice = 100
    if logged:
        maxPrice = np.log(maxPrice)
        minPrice = np.log(minPrice)

    price = normPrice * (maxPrice - minPrice) + minPrice
    if logged:
        return np.exp(price)
    return price


def processPrice(df: pd.DataFrame):
    maxPrice = 2000
    minPrice = 100
    loggedMaxPrice = np.log(maxPrice)
    loggedMinPrice = np.log(minPrice)

    def str2float(price: Union[str, None]):       
        try:
            price = float(price.replace('.', ''))
        except:
            print('ERROR: Normalize price failed for "{0}"'.format(price))
            price = -1

        if price > maxPrice or price < minPrice:
            price = -1
        
        return price

    def logNormalizePrice(price: float):
        logged = np.log(price)
        return (logged - loggedMinPrice) / (loggedMaxPrice - loggedMinPrice)

    def normalizePrice(price: float):
        return (price - minPrice) / (maxPrice - minPrice)

    df['price'] = pd.to_numeric(df['price'].apply(str2float))
    df = df[df['price'] > 0.0].copy()
    df['price'] = df['price'].apply(normalizePrice)

    print('Items after price:', len(df))

    return df

def processUsefulArea(df: pd.DataFrame):
    maxArea = 200
    minArea = 20
    loggedMaxArea= np.log(maxArea)
    loggedMinArea = np.log(minArea)

    def str2float(area: Union[str, None]):       
        try:
            area = area.replace('.', '').strip()
            area = area.replace('mp','')

            if ',' in area:
                area = area.split(',')[0]

            area = float(area)
        except:
            #print('ERROR: Normalize area failed for "{0}"'.format(area))
            area = -1

        if area > maxArea or area < minArea:
            area = -1
        
        return area

    def logNormalizeArea(price: float):
        logged = np.log(price)
        return (logged - loggedMinArea) / (loggedMaxArea - loggedMinArea)

    def normalizeArea(price: float):
        return (price - minArea) / (maxArea - minArea)

    df['useful_area'] = pd.to_numeric(df['useful_area'].apply(str2float))
    df = df[df['useful_area'] > 0.0].copy()
    df['useful_area'] = df['useful_area'].apply(normalizeArea)

    print('Items after useful_area:', len(df))

    return df

def processNRooms(df: pd.DataFrame):
    def normalizeRooms(nRooms: Union[str, None]):
        try:
            n = int(nRooms)
            if n > 6:
                n = -1
        except:
            print('ERROR: Normalize nrooms failed for "{0}"'.format(nRooms))
            n = -1

        if n >= 4:
            n = '4+'
        else:
            n = str(n)
        
        return n
    
    df['n_rooms'] = df['n_rooms'].apply(normalizeRooms).astype('string')
    df = df[df['n_rooms'] != '-1'].copy()
    df['n_rooms'] = df['n_rooms'].astype('category')
    
    print('Items after n_rooms:', len(df))

    return df

def processNBathrooms(df: pd.DataFrame):
    def normalizeBathrooms(nRooms: Union[str, None]):
        try:
            n = int(nRooms)
        except:
            n = 1

        if n >= 2:
            n = '2+'
        else:
            n = str(n)
        
        return n
    
    df['n_bathrooms'] = df['n_bathrooms'].apply(normalizeBathrooms).astype('category')
    
    print('Items after n_bathrooms:', len(df))

    return df

def processNParking(df: pd.DataFrame):
    def normalizeParking(nParking: Union[str, None]):
        if nParking is None:
            return '0'
        nParking = nParking.split(' ')[0]
        try:
            nParking = int(nParking)
        except:
            nParking = 0
        return str(nParking)
    
    df['n_parking_spaces'] = df['n_parking_spaces'].apply(normalizeParking).astype('category')
    
    print('Items after n_parking_spaces:', len(df))

    return df

def processRoomConf(df: pd.DataFrame):
    def normalizeRoomConf(roomConf: Union[str, None]):
        translateConf = {
            'decomandat': 'detached',
            'semidecomandat': 'semi-detached',
            'detached': 'detached',
            'semi-detached': 'semi-detached'
        }

        if roomConf is None:
            return 'detached'

        return translateConf.get(roomConf.strip(), 'detached')

    df['room_config'] = df['room_config'].apply(normalizeRoomConf).astype('category')

    print('Items after room_config:', len(df))

    return df

def processYear(df: pd.DataFrame):
    def getYear(year: Union[str, None]):
        if year is None:
            return -1
        tokens = year.split(' ')
        n = -1
        for tok in tokens:
            try:
                n = int(tok)
                break
            except:
                continue

        return n

    def normalizeYear(year: int):
        if year < 1977:
            return 'x<1977'
        if year < 2010:
            return '1977<=x<2000'
        return '2010<=x'    

    df['year_construction'] = pd.to_numeric(df['year_construction'].apply(getYear))
    df = df[df['year_construction'] > 0.0].copy()
    df['year_construction'] = df['year_construction'].apply(normalizeYear).astype('category')

    print('Items after year_construction:', len(df))

    return df

def preprocess(df: pd.DataFrame):

    dropCols = [
        'floor', 'n_kitchens', 'total_useful_area', 'n_garages',
        'built_area', 'building_type', 'n_balconies', 'building_material',
        'comfort', 'floors_config'
    ]

    df = df.drop(columns=dropCols)

    print('Initial items:', len(df))

    df = processPrice(df)
    df = processNRooms(df)
    df = processRoomConf(df)
    df = processUsefulArea(df)
    df = processYear(df)
    df = processNBathrooms(df)
    df = processNParking(df)

    return df
    
def getFeatures(df: pd.DataFrame):
    
    labels = np.array(df['price'], dtype='float32')

    features = []

    for key in keys:
        if key in df:
            print(df[key].head())

            if key in categorical:
                features.append(np.array(pd.get_dummies(df[key]), dtype='float32'))
            else:
                arr = np.array(df[key], dtype='float32')
                n = arr.shape[0]
                features.append(arr.reshape((n, 1)))

            print(features[-1].shape)

    features = np.concatenate(features, axis=1)

    print(features.shape)

    return labels, features

if __name__ == '__main__':
    df = getBaseDf()
    df = preprocess(df)
    features = getFeatures(df)

    """
    features = pd.get_dummies(df)

    labels = np.array(features['price'])  
    features = features.drop('price', axis=1)
    features = features.drop('ID', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    
    from sklearn.model_selection import train_test_split

    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size = 0.25, random_state = 42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    """

    """
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    predictions = (predictions * 0 + 1.0) * train_labels.mean()
    predictions = reverseNormPrice(predictions, logged=True)

    test_labels = reverseNormPrice(test_labels, logged=True)
    # Calculate the absolute errors

    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'euros.')

    print(test_labels[:10])
    print(predictions[:10])
    """

    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    mse = tf.keras.losses.MeanSquaredError()
    
    model.compile(optimizer='adam', loss=mse)

    model.fit(train_features, train_labels, epochs=5)

    model.evaluate(test_features,  test_labels, verbose=2)

    predictions = model(test_features).numpy().flatten()
    
    predictions = reverseNormPrice(predictions, logged=True)
    test_labels = reverseNormPrice(test_labels, logged=True)

    print(predictions[:10])
    print(test_labels[:10])

    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'euros.')
    """