from keras.models import Sequential
from keras.layers import Dense

def get_model(input_dim=12, output_dim=1):
    # define model
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim))
    model.summary()

    return model