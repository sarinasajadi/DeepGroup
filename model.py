import tensorflow
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Concatenate, Multiply, Dense, Activation, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2


def get_model(num_users, n_latent_factors_user, num_items, n_latent_factors_items, hidden_units, x_train, y_train, num_epochs=100, batch_size=2048, x_test=None, model_type='user'):

    input_shape = 1
    data_type = 'int32'
    if model_type == 'group':
        input_shape = n_latent_factors_user
        data_type = 'float32'

    user_input = Input(shape=(input_shape,), dtype=data_type, name=model_type + '_input')
    item_input = Input(shape=(input_shape,), dtype=data_type, name='item_input')

    if model_type == 'user':
        user_embedding = Embedding(input_dim=num_users, output_dim=n_latent_factors_user, name=model_type +'_embedding', embeddings_initializer='random_normal', embeddings_regularizer=l2(0), input_length=1)
        item_embedding = Embedding(input_dim=num_items, output_dim=n_latent_factors_items, name='item_embedding', embeddings_initializer='random_normal', embeddings_regularizer=l2(0), input_length=1)

        # Crucial to flatten an embedding vector!
        user_vec = Flatten()(user_embedding(user_input))
        item_vec = Flatten()(item_embedding(item_input))

    else:
        user_vec = user_input
        item_vec = item_input

    # concatanation or multiplication of user embedding and item embdding
    # one of them goes through hidden layers
    concatenation = Concatenate(axis=1)([user_vec, item_vec])
    # latent factors for users and items should be the same for multiplication
    multiplication = Multiply()([user_vec, item_vec])

    # hidden layers
    h_layer = Dense(hidden_units[0], activation='relu', kernel_regularizer=l2(0), activity_regularizer=regularizers.l2(0))(concatenation)
    for i in range(1, len(hidden_units)):
        h_layer = Dense(hidden_units[i], activation='relu', kernel_regularizer=l2(0), activity_regularizer=regularizers.l2(0))(h_layer)
        h_layer = Dropout(0.2, name='Dropout'+str(i))(h_layer)

    result = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(h_layer)

    adam = tensorflow.keras.optimizers.Adam(learning_rate=0.001, amsgrad=False)

    model = tensorflow.keras.Model([user_input, item_input], result)

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    #we want the input to the functional model to be a list of numpy arrays
    #y_train should also be an numpy array
    x_train = [np.array(x_train[0]), np.array(x_train[1])]
    if x_test is not None:
        x_test = [np.array(x_test[0]), np.array(x_test[1])]
    y_train = np.array(y_train)

    # The model should also be fit with validation data to prevent overfitting using callbacks like early stopping
    history = model.fit(x_train, np.array(y_train), batch_size=batch_size, epochs=num_epochs, verbose=1, shuffle=True)

    # The test data should be different from the training data (I don't know if this is just for testing but I thought I should add this)
    score = model.evaluate(x_train, y_train, verbose=1)

    print('Model test loss:', score[0])
    print('Model test accuracy:', score[1] * 100)
    print('Model training accurracy', np.mean(history.history['accuracy']) * 100)

    # I am not sure what this section does, but it will cause a crash after model.predict since x_test is empty
    # Any valid version of x_test will also need to be of the form [np.array, np.array]
    if x_test is None:
        x_test = []

    if model_type == 'user':
        weights = model.get_weights()
        user_embeddings = weights[0]
        item_embeddings = weights[1]
        return user_embeddings, item_embeddings

    else:
        y_test_pred = model.predict(x_test, verbose=1)
        return y_test_pred
