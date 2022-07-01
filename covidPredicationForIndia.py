# LSTM for international airline passengers problem with regression framing
import numpy 
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	#return dataX, dataY
	return numpy.array(dataX), numpy.array(dataY)

# load the dataset
dataset = pd.read_csv("./covid_data.csv")
indian_covid_data = dataset[72076:72867]
features = indian_covid_data['new_cases'].values

#print(features)                        type(features)        --------------------       <class 'numpy.ndarray'>

# split into train and test sets
train_size = int(len(features) * 0.7)
test_size = len(features) - train_size
train, test = features[0:train_size], features[train_size:len(features)]
#print(train)                           type(train))          --------------------       <class 'numpy.ndarray'>
#print(test)                            type(test))           --------------------       <class 'numpy.ndarray'>

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
#print(trainX)                          type(trainX)          --------------------       <class 'numpy.ndarray'>
#print(trainY)                          type(trainX)          --------------------       <class 'numpy.ndarray'>
#print(testX)                           type(testX)           --------------------       <class 'numpy.ndarray'>
#print(testY)                           type(testY)           --------------------       <class 'numpy.ndarray'>

# reshape input to be [samples, time steps, features] 
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=2, batch_size=16, verbose=2)



'''
this is needed when we perform scaling, that is if we use any scaler function to get our data scaled. here in this we don't use any such kind of methodology
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
'''
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


#print(trainPredict)
#print(testPredict)

## shift train predictions for plotting
trainPredictPlot = numpy.empty_like(features)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(features)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(features)-1, :] = testPredict

# plot baseline and predictions
plt.plot(features)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

