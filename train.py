import json


def train_model(model, X_train, y_train, X_val, y_val, epochs=200):
    # model = simple_lstm.get_model()
    # model.summary()
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_val, y_val))

    model_json = model.to_json()
    with open("checkpoint/simple_model.json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights("checkpoint/simple_model.h5")

    return model