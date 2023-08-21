import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from tensorflow.keras.utils import plot_model
import pydot
import graphviz

data = 'admissions_data.csv'
#Best R-Squared = 0.8337742714594505
class AdmissionRNN:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.input_data = self.data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
        self.labels = self.data[['Chance of Admit ']]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input_data, self.labels, test_size=0.2, random_state=42)
        self.scaler = MinMaxScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        self.model = self.build_model()
        self.history = None
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(units=7, activation='relu'))
        model.add(Dense(units=7, activation='relu'))
        model.add(Dense(units=4, activation='relu'))
        model.add(Dense(units=1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.2, random_state=42):
        self.history = self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
        
    def evaluate_model(self):
        loss, mae = self.model.evaluate(self.x_test, self.y_test)
        print("Test Loss:", loss)
        print("Mean Absolute Error:", mae)
        predictions = self.model.predict(self.x_test)
        r2 = r2_score(self.y_test, predictions)
        print("R-squared:", r2)

    def predict(self, data):
    	data = np.array(data).reshape(1,-1)
    	scaled = self.scaler.transform(data)
    	return f"Your chance of Admission is: {100 * self.model.predict(scaled).flatten()[0]:.1f}%"


    def plot_loss(self):
    	history_dict = self.history.history
    	loss = history_dict['loss']
    	val_loss = history_dict['val_loss']
    	epochs = range(0, len(loss), + 1)
    	plt.figure(figsize=(12,6))
    	plt.subplot(1,2,1)
    	plt.plot(epochs, loss, 'bo', markersize=3, label='Training Loss')
    	plt.plot(epochs, val_loss, 'orange', label = 'Validation Loss')
    	plt.title('Training and Validation Loss')
    	plt.xlabel('Epochs')
    	plt.ylabel('Loss')
    	plt.legend()
    	return plt.show()

    def plot_epoch_mae(self):
    	history_dict = self.history.history
    	loss = history_dict['loss']
    	epochs = range(1, len(loss) + 1)
    	mae = history_dict['mae']
    	val_mae = history_dict['val_mae']
    	plt.figure(figsize=(12,6))
    	plt.subplot(1,2,1)
    	plt.plot(epochs, mae, 'bo', markersize=3, label='Training MAE')
    	plt.plot(epochs, val_mae, 'orange', label='Validation MAE')
    	plt.title('Training and Validation MAE')
    	plt.xlabel('Epochs')
    	plt.ylabel('MAE')
    	plt.legend()
    	return plt.show()    

    def plot_feature_importance(self):
    	#Get the weights of the first layer
    	first_layer_weights = self.model.layers[0].get_weights()[0]
    	#Importances = Absolute value of weights 
    	feature_importance = np.abs(first_layer_weights).mean(axis=1)
    	#Feature Names
    	feature_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
    	#Plot 
    	plt.figure(figsize=(10,6))
    	plt.bar(feature_names, feature_importance)
    	plt.ylim([0, 0.5])
    	plt.yticks(np.arange(0, 0.55, 0.05))
    	plt.xlabel('Features')
    	plt.ylabel('Feature Importances')
    	plt.title('Importances of Each Feature')
    	return plt.show()

    def plot_prediction_actual(self):
    	#Change to actual - residual plot
   		predictions = self.model.predict(self.x_test)
   		actual = self.y_test.to_numpy().flatten()
   		residual = actual - predictions.flatten()
   		plt.figure(figsize=(10,6))
   		plt.scatter(actual, residual, color='blue', alpha=0.5)
   		plt.xlabel('Actual Values')
   		plt.ylabel('Residuals')
   		plt.title('Actual vs Residual')
   		plt.grid(True)
   		return plt.show()


test_prediction = [325,112,3,3.5,3.5,8.45,0]
nnmodel = AdmissionRNN(data)
nnmodel.build_model()
nnmodel.train_model()
nnmodel.evaluate_model()
#nnmodel.plot_loss()
#nnmodel.plot_epoch_mae()
#nnmodel.plot_feature_importance()
#nnmodel.plot_prediction_actual()
print(nnmodel.predict(test_prediction))



