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
import random, math
from numpy import concatenate
from math import sqrt
from six.moves import xrange
from numpy import array
from keras.utils import plot_model
from keras.layers import Dropout
from keras import backend as K
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import merge

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

	def __init__(self, config, train_dataset, eval_dataset_1, eval_dataset_2, test_dataset, real_test, real_eval_1, real_eval_2,forecasting_time,year_test,index_bairro):
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
		self.year_test = year_test
		self.index_bairro = index_bairro
		self.dropout_input = config.dropout_input
		self.clipvalue = config.clipvalue
		self.forecasting_time = forecasting_time
		self.config = config

		self.train_dataset = train_dataset
		self.eval_dataset_1 = eval_dataset_1
		self.eval_dataset_2 = eval_dataset_2
		self.test_dataset = test_dataset
		self.real_test = real_test
		self.real_eval_1 = real_eval_1
		self.real_eval_2 = real_eval_2


		self.is_test = False

		self.X = []
		self.X_emb = []
		self.y = []
		for i in range(0, len(self.train_dataset)):
			self.X.append(np.transpose(np.array([self.train_dataset[i][0]])))
			self.X_emb.append(self.train_dataset[i][1])
			self.y.append(self.train_dataset[i][2])

		self.X = np.array(self.X)
		self.X_emb = np.array(self.X_emb)
		self.y = np.array(self.y)

		self.batch_size = self.define_batch_size(len(self.y))

		self.model_eval = self.get_model(batch_size=1)
		self.model_predict = self.get_model(batch_size=1)
		self.model = self.get_model(batch_size=self.batch_size)

		self.train_defaul_weights = self.model.get_weights()
		self.eval_defaul_weights = self.model_eval.get_weights()
		self.predict_defaul_weights = self.model_predict.get_weights()
		print(self.model.summary())


	def get_model(self, batch_size):

		dengue_input = Input(batch_shape = (batch_size, self.history_length,1))
		dropout_input = Dropout(self.dropout_input)(dengue_input)

		emb_input = Input(batch_shape = (batch_size, self.history_length))
		emb = Embedding(52, self.neurons_emb, input_length=self.history_length)(emb_input)
		dropout_emb = Dropout(self.dropout)(emb)

		merged = keras.layers.concatenate([dropout_input, dropout_emb])

		lstm = LSTM(self.neurons, stateful=True,
			kernel_regularizer=L1L2(l1=self.regularization_l1, l2=self.regularization_l2))(merged)

		dropout_lstm = Dropout(self.dropout)(lstm)

		output_dense = Dense(1, activation='sigmoid', use_bias=True)(dropout_lstm)

		model_temp = Model(inputs=[dengue_input,emb_input], outputs=output_dense)

		op = None
		if self.optimizer == OpUtil.RMSPROP.value:
			op = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-6, decay=0.0)
		model_temp.compile(loss='mean_squared_error', optimizer=op)

		return model_temp

	def train_model (self):
		mses_trainning = []
		mses_eval = []

		best_mse = 10000
		best_weights = None
		best_epoch = 0

		for epoch in range(0,self.epoch):
			history = self.model.fit([self.X,self.X_emb], self.y,batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)
			self.model.reset_states()

			if epoch % 10 == 0:
				loss_train = history.history['loss'][0]
				mses_trainning.append(loss_train)
				weights = self.model.get_weights()

				sum_mse_1, predicted_data_1, mse_bairros_1 = self.new_eval(weights, self.model_eval, self.eval_dataset_1, self.real_eval_1)
				sum_mse_2, predicted_data_2, mse_bairros_2 = self.new_eval(weights, self.model_eval, self.eval_dataset_2, self.real_eval_2)

				loss_eval = (sum_mse_1 + sum_mse_2) / 2
				mses_eval.append(loss_eval)
				if loss_eval < best_mse:
					best_mse = loss_eval
					best_weights = weights
					best_epoch = epoch

				print(str(epoch)+'/'+str(self.epoch) + ' loss: ' + str(loss_train) + ' loss_eval: ' + str(loss_eval))

		return mses_trainning, mses_eval, best_weights

	def new_eval(self, weights, model_eval, examples, real):

		model_eval.set_weights(weights)
		model_eval.reset_states()

		predicted_data = np.transpose(np.array([examples[0][0]]))

		num_resta = self.forecasting_time - self.history_length

		dengue_h = np.array([np.transpose(np.array([examples[0][0]]))])

		if num_resta > 1:
			model_eval.predict([dengue_h,np.array([examples[0][1]])])
		for i in range(1,num_resta):
			if (i < num_resta-1):
				dengue_h = np.transpose(np.array([examples[i][0]]))
				model_eval.predict([np.array([dengue_h]),np.array([examples[i][1]])])
			last_line = examples[i][0][(len(examples[i][0])-1)]
			last_line = np.array([[last_line]])
			predicted_data = np.concatenate((predicted_data, last_line), axis=0)


		history = np.transpose(np.array([examples[num_resta-1][0]]))
		history = np.array([history])
		for i in range(num_resta-1, len(examples)):
			week = np.array([examples[i][1]])
			w_prediction = model_eval.predict([history, week])
			history = np.concatenate((history[:, 1:, :], np.expand_dims(w_prediction, 0)), axis=1)
			predicted_data = np.concatenate((predicted_data, w_prediction), axis=0)

		predicted_data = np.transpose(predicted_data)

		mse_bairros = []
		predictions = predicted_data[0]
		real_y = real

		predictions_temp = []
		real_y_temp = []
		for index in range(self.history_length, len(predictions)):
			predictions_temp.append(predictions[index])
			real_y_temp.append(real_y[index])

		mse = mean_squared_error(real_y_temp, predictions_temp)
		mse_bairros.append(mse)

		return mse, predicted_data, mse_bairros

	def reset_models(self):
		self.model.set_weights(self.train_defaul_weights)
		self.model.reset_states()

		self.model_eval.set_weights(self.eval_defaul_weights)
		self.model_eval.reset_states()

		self.model_predict.set_weights(self.predict_defaul_weights)
		self.model_predict.reset_states()

	def update_datasets(self,train_dataset, eval_dataset_1, eval_dataset_2, test_dataset, real_test,real_eval_1,real_eval_2,year_test,index_bairro):
		self.train_dataset = train_dataset
		self.eval_dataset_1 = eval_dataset_1
		self.eval_dataset_2 = eval_dataset_2
		self.test_dataset = test_dataset
		self.real_test = real_test
		self.real_eval_1 = real_eval_1
		self.real_eval_2 = real_eval_2
		self.year_test = year_test
		self.index_bairro = index_bairro

		self.X = []
		self.X_emb = []
		self.y = []
		for i in range(0, len(self.train_dataset)):
			self.X.append(np.transpose(np.array([self.train_dataset[i][0]])))
			self.X_emb.append(self.train_dataset[i][1])
			self.y.append(self.train_dataset[i][2])

		self.X = np.array(self.X)
		self.X_emb = np.array(self.X_emb)
		self.y = np.array(self.y)

		self.reset_models()

	def define_batch_size(self, size):
		bt = 20
		while (size % bt != 0):
			bt = bt + 1
		return bt

	def change_dropout_predict (self,weights, model_eval, examples, real):

		model_eval.set_weights(weights)
		model_eval.reset_states()

		for layer in model_eval.layers:
			if "dropout" in layer.name:
				layer.rate = random.uniform(0, 1)

		f = K.function([model_eval.layers[1].input,model_eval.layers[0].input, K.learning_phase()],
		   [model_eval.layers[-1].output])

		predicted_data = np.transpose(np.array([examples[0][0]]))

		num_resta = self.forecasting_time - self.history_length

		dengue_h = np.array([np.transpose(np.array([examples[0][0]]))])

		if num_resta > 1:
			f([dengue_h,np.array([examples[0][1]]),1])[0]

		for i in range(1,num_resta):
			if (i < num_resta-1):
				dengue_h = np.transpose(np.array([examples[i][0]]))
				f([np.array([dengue_h]),np.array([examples[i][1]]),1])[0]


			last_line = examples[i][0][(len(examples[i][0])-1)]
			last_line = np.array([[last_line]])
			predicted_data = np.concatenate((predicted_data, last_line), axis=0)


		history = np.transpose(np.array([examples[num_resta-1][0]]))
		history = np.array([history])
		for i in range(num_resta-1, len(examples)):
			week = np.array([examples[i][1]])
			x = [history,week]
			w_prediction = f([history,week,1])[0]
			history = np.concatenate((history[:, 1:, :], np.expand_dims(w_prediction, 0)), axis=1)
			predicted_data = np.concatenate((predicted_data, w_prediction), axis=0)

		predicted_data = np.transpose(predicted_data)

		mse_bairros = []
		predictions = predicted_data[0]
		real_y = real

		predictions_temp = []
		real_y_temp = []
		for index in range(self.history_length, len(predictions)):
			predictions_temp.append(predictions[index])
			real_y_temp.append(real_y[index])

		mse = mean_squared_error(real_y_temp, predictions_temp)
		mse_bairros.append(mse)

		return mse, predicted_data, mse_bairros

	def run(self):

		#train model
		mses_trainning, mses_eval, best_weights = self.train_model()

		#predict
		self.is_test = True
		sum_mse_2, predicted_test, mse_bairros_test = self.new_eval(best_weights, self.model_predict, self.test_dataset, self.real_test)



		self.model_predict.set_weights(best_weights)
		self.model_predict.reset_states()

		path_test_charts = 'models_weights/'+ str(self.config.id) + '/' + str(self.year_test)+'/'
		if not os.path.exists(path_test_charts):
			os.makedirs(path_test_charts)

		# serialize model to JSON
		model_json = self.model_predict.to_json()
		with open(path_test_charts + str(self.index_bairro) + ".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.model_predict.save_weights(path_test_charts + str(self.index_bairro) + ".h5")

		return mses_trainning, mses_eval, predicted_test, self.model_predict.summary()
