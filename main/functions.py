import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import style
import csv
from django.conf import settings
import numpy
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn import preprocessing
from keras.layers.core import Dense, Activation, Dropout
import time
from keras.models import load_model
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pickle
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.layers import Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import np_utils
import matplotlib.pyplot as plt
from matplotlib import style
from pandas import read_csv
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from keras.layers import Dense
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, precision_score
import pandas as pd
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib import style
import seaborn as sns
import numpy as np
import csv
import pandas as pd
import numpy as np
from pandas import read_csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
import math
from keras.models import load_model
from sklearn.model_selection import train_test_split
import math
import csv
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import style
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def afficher_annees_precedentes(csv_file, annee):
    sns.set()
    style.use('ggplot')

    regions = []
    flu_young = []
    sari_young = []
    cons_young = []
    flu_ad = []
    sari_ad = []
    cons_ad = []
    flu_ag = []
    sari_ag = []
    cons_ag = []
    tot_flu = []
    tot_sari = []
    tot_cons = []
    t_flu = []
    t_sari = []
    dossier = settings.STATIC_URL+"main/images/resultats_annees_precedentes"
    with open(csv_file) as csvfile:
        ilireader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(ilireader, None)  # skipping the header
        for row in ilireader:
            regions.append(row[0])
            flu_young.append(int(row[1]))
            sari_young.append(int(row[2]))
            cons_young.append(int(row[3]))
            flu_ad.append(int(row[4]))
            sari_ad.append(int(row[5]))
            cons_ad.append(int(row[6]))
            flu_ag.append(int(row[7]))
            sari_ag.append(int(row[8]))
            cons_ag.append(int(row[9]))
            tot_flu.append(int(row[10]))
            tot_cons.append(int(row[11]))
            tot_sari.append(int(row[12]))
            t_flu.append(float(row[13]))
            t_sari.append(float(row[14]))

    y_pos = np.arange(len(regions))
    plt.rcParams['figure.figsize'] = (8, 6)

    plt.subplots()
    plt.bar(regions, tot_flu, label='Total Flu', color='lightcoral', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('Total des cas de grippe')
    plt.title('Total des cas de grippe par région en Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/1totflu.png')

    plt.subplots()
    plt.bar(regions, tot_sari, label='Total Sari', color='lightskyblue', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('Total des cas SARI')
    plt.title('Total des cas SARI par région en Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/2totsari.png')

    plt.subplots()
    plt.bar(regions, tot_cons, label='Total Consultant', color='yellowgreen', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('Total des consultations')
    plt.title('Total des consultations par région en Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/3totcons.png')

    plt.subplots()
    plt.bar(regions, flu_young, label='< 5 years', color='gold', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('Grippe')
    plt.title("Total des cas de grippe par région en Tunisia pour les enfants dont l'âge est inférieur à 5 ans")
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/4fluyo09-10.png')

    plt.subplots()
    plt.bar(regions, flu_ad, label='5 - 16 years', color='magenta', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('Grippe')
    plt.title("Total des cas de grippe par région en Tunisia pour les enfants dont l'âge est entre 5 et 16 ans")
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/5fluad.png')

    plt.subplots()
    plt.bar(regions, flu_ag, label='> 16 years', color='green', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('Grippe')
    plt.title("Total des cas de grippe par région en Tunisia pour adultes")
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/6fluag.png')

    plt.subplots()
    plt.bar(regions, sari_young, label='< 5 years', color='gold', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('SARI')
    plt.title("Total des SARI par région en Tunisia pour les enfants dont l'âge est inférieur à 5 ans")
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/7sariyo.png')

    plt.subplots()
    plt.bar(regions, sari_ad, label='5 - 16 years', color='magenta', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('SARI')
    plt.title("Total des cas SARI par région en Tunisia pour les enfants dont l'âge est entre 5 et 16 ans")
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/8sariad.png')

    plt.subplots()
    plt.bar(regions, sari_ag, label='> 16 years', color='green', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('SARI')
    plt.title("Total des cas de grippe par région en Tunisia pour adultes")
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/9sariag.png')

    plt.subplots()
    plt.bar(regions, t_flu, label='Flu rate', color='lightcoral', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('Pourcentage des cas de grippe')
    plt.title('Pourcentage des cas de grippe par région en Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/9ztflu.png')

    plt.subplots()
    plt.bar(regions, t_sari, label='Sari rate', color='lightskyblue', width=0.5)
    plt.xticks(y_pos, regions, rotation=45, fontsize='10', horizontalalignment='right')
    plt.xlabel('Région')
    plt.ylabel('Pourcentage des cas SARI')
    plt.title('Pourcentage des cas SARI par région en Tunisia')
    plt.legend()
    plt.savefig(dossier+'/'+annee+'/9zztsari.png')

    liste_region_epidemiologique = []
    for val in t_flu:
        if val >= 10.0:
            ind = t_flu.index(val)
            liste_region_epidemiologique.append(regions[ind])
    return liste_region_epidemiologique


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def predire_lstm1():
    dossier = settings.STATIC_URL + "main/images/predictions"
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    cols = ['region', 'flu_young', 'sari_young', 'cons_young', 'flu_adult', 'sari_adult',
            'cons_adult', 'flu_aged', 'sari_aged', 'cons_aged', 'tot_flu', 'tot_sari',
            'tot_cons', 'week', 'month', 'year', 't_flu', 't_sari']
    df = read_csv(settings.MEDIA_URL+'csv/tot.csv', names=cols)
    df = df.iloc[::-1]
    df = df.drop(['region', 'flu_young', 'sari_young', 'cons_young', 'flu_adult', 'sari_adult',
                  'cons_adult', 'flu_aged', 'sari_aged', 'cons_aged', 'tot_flu', 'tot_sari',
                  'tot_cons', 'week', 'month', 'year', 't_sari'], axis=1)
    dataset = df.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    look_back = 3
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    plt.plot(scaler.inverse_transform(dataset), color='orchid', label='Real values')
    plt.plot(trainPredictPlot, color='gold', label='Predicted train values')
    plt.plot(testPredictPlot, color='coral', label='Predicted test values')
    plt.title('Real values vs predicted values using LSTM')
    plt.legend()
    plt.savefig(dossier + '/lstm1/lstm1.png')
    plt.show()
    erreur = [trainScore, testScore]
    return erreur


def predire_lstm4():
    dossier = settings.STATIC_URL + "main/images/predictions"
    sns.set()
    style.use('ggplot')
    # load the dataset
    cols = ['region', 'flu_young', 'sari_young', 'cons_young', 'flu_adult', 'sari_adult',
            'cons_adult', 'flu_aged', 'sari_aged', 'cons_aged', 'tot_flu', 'tot_sari',
            'tot_cons', 'week', 'month', 'year', 't_flu', 't_sari']
    df = read_csv(settings.MEDIA_URL + 'csv/tot.csv', names=cols)
    ddf = df.iloc[::-1]
    le = preprocessing.LabelEncoder()
    df['region'] = le.fit_transform(df['region'])
    df['month'] = le.fit_transform(df['month'])
    df['region'] = df['region'].apply(pd.to_numeric)
    df['flu_young'] = df['flu_young'].apply(pd.to_numeric)
    df['sari_young'] = df['sari_young'].apply(pd.to_numeric)
    df['cons_young'] = df['cons_young'].apply(pd.to_numeric)
    df['flu_adult'] = df['flu_adult'].apply(pd.to_numeric)
    df['sari_adult'] = df['sari_adult'].apply(pd.to_numeric)
    df['cons_adult'] = df['cons_adult'].apply(pd.to_numeric)
    df['flu_aged'] = df['flu_aged'].apply(pd.to_numeric)
    df['sari_aged'] = df['sari_aged'].apply(pd.to_numeric)
    df['cons_aged'] = df['cons_aged'].apply(pd.to_numeric)
    df['tot_flu'] = df['tot_flu'].apply(pd.to_numeric)
    df['tot_sari'] = df['tot_sari'].apply(pd.to_numeric)
    df['tot_cons'] = df['tot_cons'].apply(pd.to_numeric)
    df['week'] = df['week'].apply(pd.to_numeric)
    df['month'] = df['month'].apply(pd.to_numeric)
    df['year'] = df['year'].apply(pd.to_numeric)
    df['t_flu'] = df['t_flu'].apply(pd.to_numeric)
    df['t_sari'] = df.t_sari.convert_objects(convert_numeric=True)
    df['t_sari'] = df['t_sari'].apply(pd.to_numeric)
    df_list = list(df.columns)
    labels = np.array(df['t_flu'])
    df = df.drop('t_flu', axis=1)
    df = df.drop('t_sari', axis=1)
    features = np.array(df)
    # Take 80% of data as the training sample and 20% as testing sample
    trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.20, shuffle=False)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(trainX, trainY, epochs=1000, batch_size=175,
              validation_data=(testX, testY), verbose=2, shuffle=False)

    testPredict = model.predict(testX)
    # save the model
    pickle.dump(model, open(dossier+'/LSTM4.sav', 'wb'))
    # calculate root mean squared error
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test MSE score : %.2f ' % (testScore))
    # plot predictions
    plt.plot(testY, color='royalblue', label='Real values')
    plt.plot(testPredict, color='gold', label='Predicted values')
    plt.title('Real values vs predicted values using LSTM')
    plt.legend()
    plt.savefig(dossier + '/lstm4/lstm4.png')
    plt.show()
    erreur = [testScore]
    return erreur


def predire_reseaux_des_neurones():
    dossier = settings.STATIC_URL + "main/images/predictions"
    style.use('ggplot')
    sns.set(style="whitegrid", color_codes=True)
    plt.rcParams['figure.figsize'] = (8, 6)

    cols = ['region', 'flu_young', 'sari_young', 'cons_young', 'flu_adult', 'sari_adult',
            'cons_adult', 'flu_aged', 'sari_aged', 'cons_aged', 'tot_flu', 'tot_sari',
            'tot_cons', 'week', 'month', 'year', 't_flu', 't_sari']
    df = read_csv(settings.MEDIA_URL + 'csv/tot.csv', names=cols)
    print("Before drop duplicants :", df.shape)
    df.drop_duplicates(keep='first', subset=None, inplace=True)
    print("After drop duplicants :", df.shape)
    le = preprocessing.LabelEncoder()
    df['region'] = le.fit_transform(df['region'])
    df['month'] = le.fit_transform(df['month'])
    print(df['region'])
    print(df['month'])
    df['region'] = df['region'].apply(pd.to_numeric)
    df['flu_young'] = df['flu_young'].apply(pd.to_numeric)
    df['sari_young'] = df['sari_young'].apply(pd.to_numeric)
    df['cons_young'] = df['cons_young'].apply(pd.to_numeric)
    df['flu_adult'] = df['flu_adult'].apply(pd.to_numeric)
    df['sari_adult'] = df['sari_adult'].apply(pd.to_numeric)
    df['cons_adult'] = df['cons_adult'].apply(pd.to_numeric)
    df['flu_aged'] = df['flu_aged'].apply(pd.to_numeric)
    df['sari_aged'] = df['sari_aged'].apply(pd.to_numeric)
    df['cons_aged'] = df['cons_aged'].apply(pd.to_numeric)
    df['tot_flu'] = df['tot_flu'].apply(pd.to_numeric)
    df['tot_sari'] = df['tot_sari'].apply(pd.to_numeric)
    df['tot_cons'] = df['tot_cons'].apply(pd.to_numeric)
    df['week'] = df['week'].apply(pd.to_numeric)
    df['month'] = df['month'].apply(pd.to_numeric)
    df['year'] = df['year'].apply(pd.to_numeric)
    df['t_flu'] = df['t_flu'].apply(pd.to_numeric)
    df['t_sari'] = df.t_sari.convert_objects(convert_numeric=True)
    df['t_sari'] = df['t_sari'].apply(pd.to_numeric)
    labels = np.array(df['t_flu'])
    df = df.drop('t_flu', axis=1)
    df = df.drop('t_sari', axis=1)
    df_list = list(df.columns)
    features = np.array(df)
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    model = Sequential()
    model.add(Dense(1000, input_dim=16, activation="relu"))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(1))
    model.summary()
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    model.fit(X_train, Y_train, epochs=100)
    model.evaluate(X_test, Y_test)
    pred = model.predict(X_test)
    pickle.dump(model, open('NN1.sav', 'wb'))
    score = np.sqrt(mean_squared_error(Y_test, pred))
    print('Test score MSE : ', score)
    plt.plot(Y_test, color='red', label='Real values')
    plt.plot(pred, color='lightcoral', label='Predicted values')
    plt.title('Real values vs predicted values using NN')
    plt.legend()
    plt.savefig(dossier + '/reseaux des neurones/reseaux des neurones.png')
    plt.show()
    erreur = [score]
    return erreur


def predire_machine_learning():
    dossier = settings.STATIC_URL + "main/images/predictions"
    sns.set()
    style.use('ggplot')
    np.random.seed(1)

    cols = ['region', 'flu_young', 'sari_young', 'cons_young', 'flu_adult', 'sari_adult',
            'cons_adult', 'flu_aged', 'sari_aged', 'cons_aged', 'tot_flu', 'tot_sari',
            'tot_cons', 'week', 'month', 'year', 't_flu', 't_sari']
    df = read_csv(settings.MEDIA_URL + 'csv/tot.csv', names=cols)
    print("Before drop duplicants :", df.shape)
    df.drop_duplicates(keep='first', subset=None, inplace=True)
    print("After drop duplicants :", df.shape)
    le = preprocessing.LabelEncoder()
    df['region'] = le.fit_transform(df['region'])
    df['month'] = le.fit_transform(df['month'])
    print(df['region'])
    print(df['month'])
    df['region'] = df['region'].apply(pd.to_numeric)
    df['flu_young'] = df['flu_young'].apply(pd.to_numeric)
    df['sari_young'] = df['sari_young'].apply(pd.to_numeric)
    df['cons_young'] = df['cons_young'].apply(pd.to_numeric)
    df['flu_adult'] = df['flu_adult'].apply(pd.to_numeric)
    df['sari_adult'] = df['sari_adult'].apply(pd.to_numeric)
    df['cons_adult'] = df['cons_adult'].apply(pd.to_numeric)
    df['flu_aged'] = df['flu_aged'].apply(pd.to_numeric)
    df['sari_aged'] = df['sari_aged'].apply(pd.to_numeric)
    df['cons_aged'] = df['cons_aged'].apply(pd.to_numeric)
    df['tot_flu'] = df['tot_flu'].apply(pd.to_numeric)
    df['tot_sari'] = df['tot_sari'].apply(pd.to_numeric)
    df['tot_cons'] = df['tot_cons'].apply(pd.to_numeric)
    df['week'] = df['week'].apply(pd.to_numeric)
    df['month'] = df['month'].apply(pd.to_numeric)
    df['year'] = df['year'].apply(pd.to_numeric)
    df['t_flu'] = df['t_flu'].apply(pd.to_numeric)
    df['t_sari'] = df.t_sari.convert_objects(convert_numeric=True)
    df['t_sari'] = df['t_sari'].apply(pd.to_numeric)
    labels = np.array(df['t_flu'])
    df = df.drop('t_flu', axis=1)
    df = df.drop('t_sari', axis=1)
    df_list = list(df.columns)
    features = np.array(df)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    print(train_features)
    print(test_features)
    print(train_labels)
    print(test_labels)
    print(train_features.dtype)
    print(test_features.dtype)
    train_labels = train_labels.astype('float32')
    test_labels = test_labels.astype('float32')
    print(train_labels.dtype)
    print(test_labels.dtype)
    test_features = test_features.astype(int)
    train_features = train_features.astype(int)
    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    print("Accuracy R2 score for Random Forest : ", r2_score(test_labels.astype(int), predictions.astype(int)))
    plt.rcParams['figure.figsize'] = (8, 6)
    x = np.arange(len(predictions))
    plt.scatter(x, predictions, color='gold', label='Predicted ILI')
    plt.scatter(x, test_labels, color='seagreen', label='Tested ILI')
    plt.title("Tested values vs predicted values in Random Forest")
    plt.legend()
    plt.savefig(dossier + '/machine learning/machine learning1.png')
    plt.show()
    # KNearestNeighbors
    clf = KNeighborsRegressor(2)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    print("Accuracy R2 score for KNearestNeighbors : ", r2_score(test_labels.astype(int), predictions.astype(int)))
    x = np.arange(len(predictions))
    plt.scatter(x, predictions, color='gold', label='Predicted ILI')
    plt.scatter(x, test_labels, color='seagreen', label='Tested ILI')
    plt.title("Tested values vs predicted values in KNearestNeighbors")
    plt.legend()
    plt.savefig(dossier + '/machine learning/machine learning2.png')
    plt.show()
    # Decision Tree Regressor
    dt = DecisionTreeRegressor(random_state=0)
    dt.fit(train_features, train_labels)
    predictions = dt.predict(test_features)
    print("Accuracy R2 score for Decision tree : ", r2_score(test_labels.astype(int), predictions.astype(int)))
    x = np.arange(len(predictions))
    plt.scatter(x, predictions, color='gold', label='Predicted ILI')
    plt.scatter(x, test_labels, color='seagreen', label='Tested ILI')
    plt.title("Tested values vs predicted values in decision Tree")
    plt.legend()
    plt.savefig(dossier + '/machine learning/machine learning3.png')
    plt.show()
    erreur = [r2_score(test_labels.astype(int), predictions.astype(int)),
             r2_score(test_labels.astype(int), predictions.astype(int)),
             r2_score(test_labels.astype(int), predictions.astype(int))]
    return erreur


def predire_region(region):
    dossier = settings.STATIC_URL + "main/images/predictions"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    sns.set()
    style.use('ggplot')
    # fix random seed for reproducibility
    np.random.seed(7)
    dataset = []
    with open(settings.MEDIA_URL + 'csv/tot.csv') as csvfile:
        ilireader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(ilireader, None)  # skipping the header
        for row in ilireader:
            if row[0] == region:
                dataset.append(float(row[16]))
    dataset = np.asarray(dataset)
    dataset = dataset.reshape(-1, 1)
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # prepare the X and Y label
    X, y = create_dataset(dataset)
    # Take 80% of data as the training sample and 20% as testing sample
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.15, shuffle=False)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(trainX.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(trainX, trainY, epochs=500, batch_size=32)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    futurePredict = model.predict(np.asarray([[testPredict[-1]]]))
    futurePredict = scaler.inverse_transform(futurePredict)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[1:len(trainPredict) + 1, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict):len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset), color='sandybrown', label='Real values')
    plt.plot(trainPredictPlot, color='gold', label='Predicted train values')
    plt.plot(testPredictPlot, color='green', label='Predicted test values')
    plt.title('Real values vs predicted values using LSTM')
    plt.legend()
    plt.savefig(dossier + '/regions/'+region+'/'+region+'.png')
    plt.show()
    erreur = [testScore, trainScore]
    return erreur

