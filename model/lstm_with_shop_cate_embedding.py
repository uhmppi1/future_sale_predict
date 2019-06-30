from keras.models import Model, Input
from keras.layers import Dense, LSTM, Embedding, Concatenate
from model.repeat_layer import RepeatEmbedding

def get_model(hidden_size=20, x3_shape=(12,1), shop_num=50, cate_num=100, embed_size=2, output_dim=1):
    # define model
    input_seq_len = x3_shape[0]

    encoder_inputs_1 = Input(shape=(1,))
    encoder_inputs_2 = Input(shape=(1,))
    encoder_inputs_3 = Input(shape=x3_shape)

    shop_emb = Embedding(input_dim=shop_num, output_dim=embed_size)
    shop_embedding = shop_emb(encoder_inputs_1)
    shop_embeddings = RepeatEmbedding(input_seq_len)(shop_embedding)

    cate_emb = Embedding(input_dim=cate_num, output_dim=embed_size)
    cate_embedding = cate_emb(encoder_inputs_2)
    cate_embeddings = RepeatEmbedding(input_seq_len)(cate_embedding)

    input_concatenated = Concatenate(axis=2)([encoder_inputs_3, shop_embeddings, cate_embeddings])

    lstm_output = LSTM(hidden_size, activation='relu')(input_concatenated)
    model_output = Dense(output_dim)(lstm_output)

    model = Model([encoder_inputs_1, encoder_inputs_2, encoder_inputs_3], model_output)
    model.summary()

    return model