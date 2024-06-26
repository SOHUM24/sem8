from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

vocab=imdb.get_word_index()
class_names=['Negative', 'Positive']

reverse_index = dict([(value, key) for (key, value) in vocab.items()])
def decode(review):
  text=""
  for i in review:
    text=text+reverse_index[i]
    text=text+" "
  return text

print(decode(x_train[1]))

x_train=pad_sequences(x_train, value=vocab['the'], padding='post', maxlen=256)
x_test=pad_sequences(x_test, value=vocab['the'], padding='post', maxlen=256)

model=Sequential()
model.add(Embedding(10000,16))
model.add(GlobalAveragePooling1D())
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=4, batch_size=128, verbose=1,validation_data=(x_test, y_test))


predicted_value=model.predict(np.expand_dims(x_test[10], 0))
print(predicted_value)
if predicted_value>0.5:
  final_value=1
else:
  final_value=0
print(final_value)
print(class_names[final_value])

loss, accuracy = model.evaluate(x_test, y_test)
print("Loss :",loss)
print(f"Accuracy (Test Data) : {round(accuracy*100)}%")
