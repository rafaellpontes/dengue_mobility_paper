from pandas import DataFrame
import numpy as np
from pandas import Series
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers.merge import concatenate as concat_keras
from keras.layers import RepeatVector
import keras
from numpy import concatenate
from math import sqrt
from six.moves import xrange
from numpy import array
from keras.utils import plot_model
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import merge
from keras.utils import plot_model
from keras import backend as K
import os


from random import randint
from numpy import array
from math import ceil
from math import log10
from numpy import argmax
import glob
from sklearn import preprocessing
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
import gc
from keras.regularizers import L1L2
from util.OpUtil import OpUtil

class RNN_Model():

	def __init__(self, config, train_dataset, eval_dataset_1, eval_dataset_2, test_dataset, real_test, real_eval_1, real_eval_2,forecasting_time,year_predict):
		self.optimizer = config.optimizer
		self.dropout = config.dropout
		self.history_length = config.history_length
		self.target_length = config.target_length
		self.neurons = config.neurons
		self.batch_size = config.batch_size
		self.regularization_l1 = config.regularization_l1
		self.regularization_l2 = config.regularization_l2
		self.epoch = config.epoch
		self.learning_rate = config.learning_rate
		self.neurons_emb = config.neurons_emb
		self.dropout_input = config.dropout_input
		self.clipvalue = config.clipvalue
		self.config = config

		self.train_dataset = train_dataset
		self.eval_dataset_1 = eval_dataset_1
		self.eval_dataset_2 = eval_dataset_2
		self.test_dataset = test_dataset
		self.real_test = real_test
		self.real_eval_1 = real_eval_1
		self.real_eval_2 = real_eval_2
		self.forecasting_time = forecasting_time
		self.year_predict = year_predict

		self.is_test = False


	def load_ids(self):
		id_nome_bairro = {}
		index_id = {}
		id_list = []
		index = 0
		with open('dados/bairro_id.csv','r') as f:
		    lines = f.readlines()
		    for line in lines:
		    	fields = line.split(",")
		    	id_nome_bairro[int(fields[1])] = fields[0]
		    	index_id[index] = int(fields[1])
		    	id_list.append(int(fields[1]))
		    	index = index + 1

		return id_nome_bairro, index_id, id_list

	def extract_id_from_time_series(self,time_series):
		id_bairro = None
		new_time_series = []
		for i in range(0,len(time_series)):
			if i == 0:
				id_bairro = int(time_series[i])
			else:
				new_time_series.append(float(time_series[i]))
		return id_bairro, np.array(new_time_series)


	def get_model(self, batch_size):

		dengue_input = Input(batch_shape = (batch_size, self.history_length,119))
		dropout_input = Dropout(self.dropout_input)(dengue_input)

		emb_input = Input(batch_shape = (batch_size, self.history_length))
		emb = Embedding(52, self.neurons_emb, input_length=self.history_length)(emb_input)
		dropout_emb = Dropout(self.dropout)(emb)

		dense_emb = Dense(self.neurons_emb, use_bias=True)(dropout_emb)
		dropout_emb_dense = Dropout(self.dropout)(dense_emb)

		emb_input_transp = Input(batch_shape = (batch_size, self.history_length,(119*15)))
		emb_input_transp_dropout = Dropout(self.dropout_input)(emb_input_transp)

		lstm_transp = LSTM(100, stateful=True, return_sequences=True,
			kernel_regularizer=L1L2(l1=self.regularization_l1, l2=self.regularization_l2))(emb_input_transp_dropout)
		dropout_lstm_transp = Dropout(self.dropout)(lstm_transp)

		merged = keras.layers.concatenate([dropout_input, dropout_emb_dense, dropout_lstm_transp])

		lstm = LSTM(self.neurons, stateful=True,return_sequences=True,
			kernel_regularizer=L1L2(l1=self.regularization_l1, l2=self.regularization_l2))(merged)
		dropout_lstm = Dropout(self.dropout)(lstm)


		lstm_2 = LSTM(self.neurons, stateful=True,return_sequences=True,
			kernel_regularizer=L1L2(l1=self.regularization_l1, l2=self.regularization_l2))(dropout_lstm)
		dropout_lstm_2 = Dropout(self.dropout)(lstm_2)

		lstm_3 = LSTM(self.neurons, stateful=True,
			kernel_regularizer=L1L2(l1=self.regularization_l1, l2=self.regularization_l2))(dropout_lstm_2)
		dropout_lstm_3 = Dropout(self.dropout)(lstm_3)

		output_dense = Dense(119, activation='sigmoid', use_bias=True)(dropout_lstm_3)

		model_temp = Model(inputs=[dengue_input,emb_input,emb_input_transp], outputs=output_dense)

		op = None
		if self.optimizer == OpUtil.RMSPROP.value:
			op = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-6, decay=0.0)
		model_temp.compile(loss='mean_squared_error', optimizer=op)

		return model_temp

	def train_model (self):

		X = []
		X_emb = []
		y = []
		X_transp = []
		for i in range(0, len(self.train_dataset)):
			X.append(self.train_dataset[i][0][0])
			X_emb.append(self.train_dataset[i][1][0])
			y.append(self.train_dataset[i][2][0][0])
			X_transp.append(self.train_dataset[i][3][0])

		X = np.array(X)
		X_emb = np.array(X_emb)
		y = np.array(y)
		X_transp = np.array(X_transp)

		model_eval = self.get_model(batch_size=1)
		model = self.get_model(batch_size=self.batch_size)
		print(model.summary())

		mses_trainning = []
		mses_eval = []

		best_mse = 10000
		best_weights = None
		best_epoch = 0

		for epoch in range(0,self.epoch):
			history = model.fit([X,X_emb,X_transp], y,batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)
			model.reset_states()

			#if epoch % 10 == 0:
			loss_train = history.history['loss'][0]
			mses_trainning.append(loss_train)
			weights = model.get_weights()

			sum_mse_1, predicted_data_1, mse_bairros_1 = self.new_eval(weights, model_eval, self.eval_dataset_1, self.real_eval_1, False)
			sum_mse_2, predicted_data_2, mse_bairros_2 = self.new_eval(weights, model_eval, self.eval_dataset_2, self.real_eval_2, False)

			loss_eval = (sum_mse_1 + sum_mse_2) / (119*2)
			mses_eval.append(loss_eval)
			if loss_eval < best_mse:
				best_mse = loss_eval
				best_weights = weights
				best_epoch = epoch

			print(str(epoch)+'/'+str(self.epoch) + ' loss: ' + str(loss_train) + ' loss_eval: ' + str(loss_eval))

		return mses_trainning, mses_eval, best_weights

	def new_eval(self, weights, model_eval, examples, real, save_model):
		model_eval.set_weights(weights)
		model_eval.reset_states()

		predicted_data = np.squeeze(examples[0][0], axis=0)
		num_resta = self.forecasting_time - self.history_length

		model_eval.predict([examples[0][0],np.array([examples[0][1][0]]),np.array([examples[0][3][0]])])
		for i in range(1,num_resta):
			if (i < num_resta-1):
				model_eval.predict([examples[i][0],np.array([examples[i][1][0]]),np.array([examples[i][3][0]])])
			last_line = examples[i][0][0][len(examples[0][0][0])-1]
			last_line = np.array([last_line])
			predicted_data = np.concatenate((predicted_data, last_line), axis=0)

		history = examples[num_resta-1][0]
		for i in range(num_resta-1, len(examples)):
			w_prediction = model_eval.predict([history,np.array([examples[i][1][0]]),np.array([examples[i][3][0]])])
			history = np.concatenate((history[:, 1:, :], np.expand_dims(w_prediction, 0)), axis=1)
			predicted_data = np.concatenate((predicted_data, w_prediction), axis=0)

		predicted_data = np.transpose(predicted_data)

		sum_mse = 0.0
		mse_bairros = []
		for i in range(0,len(self.id_list)):
			predictions = predicted_data[i]
			real_y = real[i]

			predictions_temp = []
			real_y_temp = []
			for index in range(self.history_length, len(predictions)):
				predictions_temp.append(predictions[index])
				real_y_temp.append(real_y[index])

			mse = mean_squared_error(real_y_temp, predictions_temp)
			mse_bairros.append(mse)
			sum_mse = sum_mse + mse

		if save_model:
			model_eval.reset_states()
			path_test_charts = 'models_weights/'+ str(self.config.id) + '/' + str(self.year_predict)+'/'
			if not os.path.exists(path_test_charts):
				os.makedirs(path_test_charts)

			# serialize model to JSON
			model_json = model_eval.to_json()
			with open(path_test_charts + "model.json", "w") as json_file:
				json_file.write(model_json)
			# serialize weights to HDF5
			model_eval.save_weights(path_test_charts + "model.h5")

		return sum_mse, predicted_data, mse_bairros

	def get_batch_size(self):
		size = len(self.train_dataset)
		bt = 15
		while (size % bt != 0):
			bt = bt + 1
		return bt


	def run(self):
		self.id_nome_bairro, self.index_id, self.id_list = self.load_ids()

		if self.batch_size == -1:
			self.batch_size = len(self.train_dataset)
		else:
			self.batch_size = self.get_batch_size()



		#train model
		mses_trainning, mses_eval, best_weights = self.train_model()

		#predict
		model_predict = self.get_model(batch_size=1)

		self.is_test = True
		sum_mse_2, predicted_test, mse_bairros_test = self.new_eval(best_weights, model_predict, self.test_dataset, self.real_test, True)

		return mses_trainning, mses_eval, predicted_test, mse_bairros_test, model_predict.summary()
