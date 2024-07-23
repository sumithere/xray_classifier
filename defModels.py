import tensorflow as tf
from tensorflow.keras.applications import InceptionV3,VGG16,DenseNet121, ResNet50
from tensorflow.keras.models import Model
from keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Dropout
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from keras import backend as K
from numpy import dstack
from tensorflow.keras.layers import Input, LSTM, Embedding, Dot, Softmax, Multiply, Lambda

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def inceptionV3(input_layer):
    base_model = InceptionV3(input_shape=(224,224,3),
                        weights='imagenet',
                        include_top=False)

    for layer in base_model.layers[:10]:
        layer.trainable = False
        
    # x = base_model.output
    x = base_model(input_layer)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    # print(x,"flatten output")
    dense = Dense(1024, activation='relu')(x)

    return dense
    # Model1 = Model(input=base_model.inputs,output=x)
    # Model1.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #           metrics=['accuracy'])
    # return Model1

def vgg(input_layer):
    base_model = VGG16(input_shape=(224,224,3),
                        weights='imagenet',
                        include_top=False)

    for layer in base_model.layers[:10]:
        layer.trainable = False
    
    x = base_model(input_layer)
    # x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    # print(x,"flatten output")
    # 
    dense = Dense(1024, activation='relu')(x)

    return dense
    # Model2 = Model(input=base_model.inputs,output=x)
    # Model2.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #           metrics=['accuracy'])
    # return Model2

def resNet(input_layer):
    base_model = ResNet50(input_shape=(224,224,3),
                        weights='imagenet',
                        include_top=False)

    for layer in base_model.layers[:10]:
        layer.trainable = False
        
    # x = base_model.output
    x = base_model(input_layer)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    # print(x,"flatten output")

    # Model3 = Model(input=base_model.inputs,output=x)
    # Model3.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #           metrics=['accuracy'])
    # return Model3
    dense = Dense(1024, activation='relu')(x)
    return dense
    
def denseNet(input_layer):
    base_model = DenseNet121(input_shape=(224,224,3),
                        weights='imagenet',
                        include_top=False)

    for layer in base_model.layers[:10]:
        layer.trainable = False
        
    # x = base_model.output
    x = base_model(input_layer)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    # print(x,"flatten output")
    dense = Dense(1024, activation='relu')(x)

    return dense
    # Model5 = Model(input=base_model.inputs,output=x)
    # Model5.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #           metrics=['accuracy'])
    # stack_model.save('stack_model.h5')




def attention_layer(inputs):
    # Compute attention scores
    e = Dense(1, activation='tanh')(inputs)
    # Compute attention weights
    alpha = Softmax()(e)
    # Apply attention weights to inputs
    context_vector = Multiply()([inputs, alpha])
    # Sum the context vector to get the attention output
    attention_output = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
    return attention_output
    
def baseModel():
    models = [vgg,resNet,inceptionV3,CNN,denseNet]
    input_layer = Input(shape = (224, 224, 3))
    results=[]
    for model in models:
        output = model(input_layer)
        # print(output)
        results.append(output)
    merged = tf.keras.layers.Concatenate()(results)
    print(merged.shape)
    # merged = attention_layer(merged)
    x = BatchNormalization()(merged)
    x = Dense(1024,activation = 'relu')(x)
    x = Dense(512,activation = 'relu')(x)
    x = Dense(256,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(14, activation = 'softmax')(x)
    # tf.print(x,"-------------------------------------------------------------")
    stacked_model = tf.keras.models.Model(inputs = input_layer, outputs = x)
    # stack_model.save('stack_model.h5')
    return stacked_model


def CNN(input_layer):
    
    x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3))(input_layer)
    x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    # x = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), activation='relu')(x)
    # x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    dense = Dense(1024, activation='relu')(x)
    
    
    
    
    # Model4 = Sequential([
    #         Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    #         Conv2D(16, kernel_size=(3, 3), activation='relu'),
    #         MaxPooling2D(pool_size=(2, 2)),
    #         Conv2D(32, kernel_size=(5, 5), activation='relu'),
    #         Conv2D(32, kernel_size=(3, 3), activation='relu'),
    #         MaxPooling2D(pool_size=(4, 4)),
    #         Conv2D(64, kernel_size=(3, 3), activation='relu'),
    #         Conv2D(64, kernel_size=(5, 5), activation='relu'),
    #         MaxPooling2D(pool_size=(2, 2)),
    #         Conv2D(128, kernel_size=(5, 5), activation='relu'),
    #         Conv2D(128, kernel_size=(3, 3), activation='relu'),
    #         MaxPooling2D(pool_size=(2, 2)),
    #         Flatten()
    #         # Dense(14, activation='sigmoid')
    #     ])
    # x = Model4(input_layer)
    # Model4.layers[-1].output
    # print(x,"flatten output")

    # Model4.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #           metrics=['accuracy'])
    # return Model4
    # dense = Dense(1024, activation='relu')(x)
    
    
    
    return dense




# def load_all_models(n_models):
#     all_models = list()
#     # models = [vgg,resNet,inceptionV3,CNN]
#     for i in range(n_models):
#         # Specify the filename
#         # dependencies = {'f1_m': f1_m }
#         filename = '/content/model' + str(i + 1) + '.h5'
#         # load the model 
#         # model=models[i]()
#         model = load_model(filename,custom_objects=dependencies)
#         # Add a list of all the weaker learners
#         all_models.append(model)
#         print('>loaded %s' % filename)
#     return all_models

    
    
# def stacked_dataset(members, inputX):
# 	stackX = None
# 	for model in members:
# 		# make prediction
# 		yhat = model.predict(inputX, verbose=0)
# 		# stack predictions into [rows, members, probabilities]
# 		if stackX is None:
# 			stackX = yhat #
# 		else:
# 			stackX = dstack((stackX, yhat))
# 	# flatten predictions to [rows, members x probabilities]
# # 	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
# 	return stackX

# def fit_stacked_model(inputX, inputy, model):
# 	# create dataset using ensemble
# 	# fit the meta learner
#     # filename = '/content/model/stack_model.h5'
#     # load the model 
#     # dependencies = {'f1_m': f1_m }
#     # model = load_model(filename,custom_objects=dependencies)
# 	#meta learner
#     model.fit(stackedX, inputy)
# 	# return model
