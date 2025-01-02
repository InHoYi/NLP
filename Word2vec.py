import numpy as np
import bow
import tensorflow as tf

# Word2Vec (CBOW) 학습용 데이터셋
dataset = [
    "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
    "i", "love", "deep", "learning", "and", "natural", "language", "processing",
    "word2vec", "is", "a", "technique", "for", "learning", "word", "embeddings",
    "machine", "learning", "is", "a", "subfield", "of", "artificial", "intelligence",
    "data", "science", "involves", "statistics", "and", "programming",
    "artificial", "intelligence", "is", "transforming", "the", "world"
]

encodedDataset = bow.oneHotEncoding(dataset)
trainset = list(encodedDataset.values())

cbow_x = []
cbow_y = []

for i in range(len(trainset) - 2):
	cbow_x.append([trainset[i], trainset[i + 2]])
	cbow_y.append(trainset[i + 1])

cbow_x = np.array(cbow_x)
cbow_y = np.array(cbow_y)

front_input = np.array([item[0] for item in cbow_x])
back_input = np.array([item[1] for item in cbow_x])

frontwordInput = tf.keras.layers.Input(shape=(35,))
backwordInput = tf.keras.layers.Input(shape=(35,))
added = tf.keras.layers.Average()([frontwordInput, backwordInput])

hiddenLayer = tf.keras.layers.Dense(64)(added)
output = tf.keras.layers.Dense(35, activation='softmax')(hiddenLayer)

model = tf.keras.Model(inputs=[frontwordInput, backwordInput], outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.summary()

model.fit(x = [front_input, back_input], y = y, epochs = 3000)