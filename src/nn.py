import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import class_weight
from keras.models import model_from_json
from keras.utils import to_categorical
from gensim.models import Word2Vec

class NeuralNetwork():
    def __init__(self, Search):
        choice = int(input("Do you want to:\n"
                           "1. Train the Word Model\n"
                           "2. Load it from the file\n"))
        if choice == 1:
            self.__create_wordEmbed_model()
        else:
            self.__wordmodel =Word2Vec.load('models\\model.bin')
        self.__movies = Search.movies
        self.__ratings = Search.ratings
        self.__genres = Search.all_genres

    def __create_wordEmbed_model(self):
        # define training data
        titles = self.__movies.title.tolist()
        for index, title in enumerate(titles):
            titles[index] = title.lower().split()
        self.__wordmodel = Word2Vec(titles, min_count=1, size=100)
        self.__wordmodel.save('model.bin')
        return

    def __myonehot(self, movie_cats):
        # cats = get_genres(movies)
        mymovie = [0.] * len(self.__genres)
        for i, cat in enumerate(self.__genres):
            if cat in movie_cats:
                mymovie[i] = 1.
        return mymovie

    def __vectorize(self, titles):
        titles = titles.split()
        vec = np.zeros(100)
        count = 0
        for title in titles:
            vec += np.array(self.__wordmodel[title.lower()])
            count += 1
        vec /= count
        return vec

    def __create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(121, activation='relu', kernel_initializer='normal'),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
            # tf.keras.layers.Dense(11, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.__model = model


    def predict(self, userId, title, genres):
        xnew = list(self.__vectorize(title))
        genres = list(self.__myonehot(genres))
        xnew.extend(genres)
        xnew.append(userId)
        xnew = np.array([np.array(xnew)])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        xnew = scaler.fit_transform(xnew)
        xnew = tf.cast(xnew, tf.float32)
        score = np.argmax(self.__model.predict(xnew), axis=-1)
        return score[0]/2.

    def __savemodel(self):
        model_json = self.__model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.__model.save_weights("model.h5")
        print("Saved model to disk")

    def loadmodel(self):
        # load json and create model
        json_file = open('models\\model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("models\\model.h5")
        print("Loaded model from disk")
        loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.__model = loaded_model

    def evalmodel(self, x_test, y_test):
        # evaluate loaded model on test data
        self.__model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        score = self.__model.evaluate(x_test, y_test, verbose=1)
        print("%s: %.2f%%" % (self.__model.metrics_names[1], score[1] * 100))

    def train(self, x, y, ylabels, size):
        self.__create_model()
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(ylabels),
                                                          ylabels)

        class_weights = {id: value for id, value in enumerate(class_weights)}
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                x, y
            ))
        train_dataset = dataset.shuffle(size, reshuffle_each_iteration=True).batch(10)

        self.__model.fit(train_dataset, epochs=50, verbose=1, batch_size=10)

    def setupdata(self):
        data = []
        movies = self.__movies.rename(columns={"movieid": "movieId"})
        full = pd.merge(self.__ratings, self.__movies, on='movieId')
        full.sort_values(by='rating')
        full.genres = full.genres.str.split('|')
        size = len(full.userId.values)
        full.title = full.title.apply(self.__vectorize)

        full.genres = full.genres.apply(self.__myonehot)

        full[['userId', 'genres', 'rating']] = full[['userId', 'genres', 'rating']].apply(np.array)
        specific = full[['title', 'genres', 'userId', 'rating']].copy()
        target = specific.pop('rating')

        a = np.array(specific.title.tolist())
        b = np.array(specific.genres.tolist())
        c = np.array(specific.userId.tolist())
        c = c.reshape(c.shape[0], 1)
        k = np.concatenate((a, b, c), axis=1)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        k = scaler.fit_transform(k)

        y = 2 * (target.values) - 1.

        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)
        dummy_y = to_categorical(encoded_y)

        return size, k, dummy_y, y