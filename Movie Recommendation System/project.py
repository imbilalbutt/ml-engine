#* Submitted By:
#* Bilal Ahmad Butt (L15-4208)
#* Asad ur Rehman (L15-4132)
#* Movie Recommendation System
from keras.models import load_model
#from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
#from keras.models import Model
import matplotlib.pyplot as plt  
from keras import backend as K
from keras import layers
#import keras.layers
import numpy as np
import numpy
#from sklearn.model_selection import train_test_split

#from sklearn.model_selection import train_test_split
#reading whole dataset
#ratings = pd.read_csv("ratings.csv",sep='\t',names="UserID,MovieID,Rating,Timestamp".split(","))
##ratings = pd.read_csv("ratings.csv")
#1000209
ratings = np.loadtxt("ratings.csv", delimiter=",", dtype=np.intp)
ratings = ratings[1:]

#train, test = train_test_split(ratings, test_size=0.2)
##train, test = train_test_split(ratings, test_size=0.1)

#1000209

train = numpy.asarray(ratings[0:999]) #999999
test = numpy.asarray(ratings[999:1200])
#getting row count and columns count
#(rows,cols) = ratings.shape
#totalUsers = len(ratings.UserID.unique()) 
#totalMovies = len(ratings.MovieID.unique())
#print(rows)
#print(cols)

# split into input (X) and output (Y) variables
#X = ratings[['UserID','MovieID','Timestamp']]
#Selecting Only one column: Rating
#Y = ratings[['Rating']] 
#df1 = df[['a','b']]


model = Sequential()
#model.add(Dense(hiddenLayerWithTwelveNeurons=12, inputLayerNeurons=3, initializeAllWeightsWith='uniform', activationFunction='sigmoid'))
model.add(Dense(10, input_dim=2, init='uniform', activation='linear'))
#model.add(Dropout(0.2))
model.add(Dense(6, init='uniform', activation='linear'))
#model.add(Dropout(0.2))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#model.summary()

# Compile model
# configure our learning process
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])


print('\nTraining Data......')
# validation data = on which to evaluate the loss and any model metrics 
# at the end of each epoch. The model will not be trained on this data.

# predict = We can iterate on our training data in batches
# num of epochs means how many times you go through your training set
# Train the model, the number of training examples in one forward/backward pass. 
history = model.fit(train[:,0:2] ,train[:,2], epochs=3, batch_size=70, verbose=1, validation_data = (train[:,0:2] ,train[:,2]))
#model.summary()
print('\nDone training.\n')
# Generates output predictions for the input samples.
predictions = model.predict(test[:,0:2] ,verbose=0, steps=None)

#count = 0;
#for i in predictions:
#    print(i, '\n')
#    count = count+1;
    
    
#print('count = ', count);
#print('\ntou')

#output_layer = model.layers[4].output 
#output_fn = theano.function([model.layers[0].input], output_layer)

u = model.layer.get_output_at(2)
print('u = ',u)
x = train[900:953:,0:2]
#from keras import backend as K
#with a Sequential model

#from keras import backend as K
#output_layer = model.layers[2].output
#get_3rd_layer_output = K.function([model.layers[1].input], [output_layer])


#from keras import backend as K
#get_3rd_layer_output = K.function([model.layers[0].input],
#[model.layers[2].output])
#layer_output = get_3rd_layer_output([train[:,0:2]])[0:4]
#layer_output = get_3rd_layer_output(x)[0:4]
#print('get_3rd_layer_output' , layer_output)
#print ('por')


#count = 0
#for i in range(len(test)):
#    if(prd[i]==test[i][2]):
#        count+=1
#
#print(len(test),count)

#x = train[1:53:,0:2] #correct
#y = model.predict_generator(x,steps= None)
#print('y = ' ,y)

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output = get_3rd_layer_output([x])[0]

print('layer_output = ' , layer_output)

#---------------SAve & Load Model--------
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# returns a compiled model
# identical to the previous one
Loadedmodel = load_model('my_model.h5')

# save as JSON
#json_string = model.to_json()

# model reconstruction from JSON:
#from keras.models import model_from_json
#model = model_from_json(json_string)
#---------------------------------------


#y_score= model.get_output(train=False)
#x_test=model.get_input(train=False)

scores = model.evaluate(test[:,0:2],test[:,2])
#print("\nTest-Accuracy:",numpy.mean(results.history["val_acc"])*100)
print('\nTest Loss Value: ',scores[0])
print('\nTest Accuracy: ',scores[1]*100)



# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()