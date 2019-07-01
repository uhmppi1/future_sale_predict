import json
# from keras import metrics
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from time import strftime

def train_model(model, X_train, y_train, X_val, y_val, epochs=200):
    # model = simple_lstm.get_model()
    # model.summary()
    # 20190701 clipnorm 1.->0.01
    model.compile(optimizer=Adam(lr=0.0001, clipnorm=0.01), loss='mse', metrics=['mse'])

    # add call backs
    early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    model_json = model.to_json()
    curtime = strftime("%Y%m%d_%H%M")
    with open("checkpoint/%s_%s.json" % (model.name, curtime), "w") as json_file:
        json.dump(model_json, json_file)
        model_saver = ModelCheckpoint(filepath=("checkpoint/%s_%s.h5" % (model.name, curtime)),
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True)

        # fit model
        model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_val, y_val),
                    callbacks=[early_stopper, model_saver])

        model.save_weights("checkpoint/%s_%s_final.h5" % (model.name, curtime))

    return model