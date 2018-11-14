import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
# load clean data
train= pd.read_csv("processed/train.csv")
test= pd.read_csv("processed/test.csv")

y_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[y_classes].values
list_sentences_train = train["comment_text"].fillna("_na_").values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list_sentences_train)
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_train = sequence.pad_sequences(list_tokenized_train, maxlen=100)
X_test = sequence.pad_sequences(list_tokenized_test, maxlen=100)


#build model
def build_model():
    embed_size = 128
    inp = Input(shape=(100, ))
    x = Embedding(20000, embed_size)(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(60, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = build_model()
batch_size = 64
model.fit(X_train, y, epochs=3, batch_size=batch_size, 
          shuffle=True, validation_split=0.1, verbose=1)

y_test = model.predict(X_test)

#model.save('my_model.h5')

sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("baseline.csv", index=False)





