import jieba
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.models import load_model

train_df = pd.read_csv('Train_DataSet.csv', header=0)
test_df = pd.read_csv('Test_DataSet.csv', header=0)
train_lable = pd.read_csv('Train_DataSet_Label.csv', header=0)

train_df['title'] = train_df['title'].astype(str)
train_df['content'] = train_df['content'].astype(str)
test_df['title'] = test_df['title'].astype(str)
test_df['content'] = test_df['content'].astype(str)

Phrase_list = train_df['title'].str.cat(train_df['content'], sep='-')
X_test = test_df['title'].str.cat(test_df['content'], sep='-')
print(Phrase_list)

Phrase_list1 = []
X_test1 = []
k = 0
for i in Phrase_list:
    seg_list = jieba.cut(i, cut_all=True)
    seg_list = " ".join(seg_list)
    Phrase_list1.append(seg_list)
    k = k + 1

k = 0
for i in X_test:
    seg_list = jieba.cut(i, cut_all=True)
    seg_list = " ".join(seg_list)
    X_test1.append(seg_list)
    k = k + 1
k = 0

Phrase_list1 = pd.Series(Phrase_list1)
X_test1 = pd.Series(X_test1)

glove_dir = r'...\vocab.txt'

embeddings_index = {}
f = open(glove_dir, encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asanyarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 300

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

max_features = max_words

model = Sequential()
model.add(Embedding(max_features, 300))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32, activation='relu',
               dropout=0.1,
               ))
model.add(Dense(3, activation='softmax'))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=37,
                    batch_size=128,
                    validation_split=0.1
                    )

model.save('LSTM_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
model = load_model('LSTM_model.h5')
y = model.predict_classes(X_test)

ID = test_df['id']


def write_preds(ID, fname):
    pd.DataFrame({"PhraseId": ID, "Sentiment": y}).to_csv(fname, index=False, header=True)


write_preds(ID, "sub_svm_baseline.csv")
