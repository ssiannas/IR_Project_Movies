import pandas as pd
import numpy as np
import src.config as cfg
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight
from keras.models import model_from_json
from keras.utils import to_categorical
from gensim.models import Word2Vec

class NeuralNetwork():
    def __init__(self, Search):
        self.__movies = Search.movies
        self.__ratings = Search.ratings
        self.__genres = Search.all_genres
        choice = int(input("Do you want to:\n"
                           "1. Load Word Model from the file\n"
                           "2. Train the Word Model\n"))
        if choice == 2:
            self.__create_wordEmbed_model()
        else:
            self.__wordmodel =Word2Vec.load('models\\model.bin')


    def __create_wordEmbed_model(self):
        # define training data
        titles = self.__movies.title.tolist()
        for index, title in enumerate(titles):
            titles[index] = title.replace('(', '')
            titles[index] = titles[index].replace(')', '')
            titles[index] = titles[index].lower().split()
        self.__wordmodel = Word2Vec(titles, min_count=1, size=100)
        self.__wordmodel.save('models\\model.bin')
        return

    def __onehot(self,userids):
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit_transform(userids)

        return onehot_encoded

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
            tf.keras.layers.Dense(791, activation='relu'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(350, activation='relu'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.__model = model


    def predict(self, userId, title, genres):
        xnew = list(self.__vectorize(title))
        genres = list(self.__myonehot(genres))
        userId = np.atleast_1d(userId)
        userId = userId.reshape(-1,1)
        xuserid = self.__my_encoder.fit_transform(userId)
        xnew.extend(genres)
        xnew.extend(xuserid[0])
        xnew = np.array([np.array(xnew)])
        xnew = tf.cast(xnew, tf.float32)
        score = np.argmax(self.__model.predict(xnew), axis=-1)
        return (score[0]+1)/2.

    def __savemodel(self):
        model_json = self.__model.to_json()
        with open("models\\model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.__model.save_weights("models\\model.h5")
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
        self.__set_users()
        self.__my_encoder = OneHotEncoder(sparse=False,handle_unknown='ignore',categories=self.__my_users)

    def evalmodel(self, x_test, y_test):
        # evaluate loaded model on test data
        score = self.__model.evaluate(x_test, y_test, verbose=1)
        print("%s: %.2f%%" % (self.__model.metrics_names[1], score[1] * 100))

    def train(self, x, y, ylabels, size):
        self.__create_model()
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(ylabels),
                                                          ylabels)

        class_weights = {id: value for id, value in enumerate(class_weights)}
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                x, y
            ))
        train_dataset = dataset.shuffle(size, reshuffle_each_iteration=True).batch(cfg.nn_batch_size)

        self.__model.fit(train_dataset, epochs=cfg.nn_epochs, verbose=1, batch_size=cfg.nn_batch_size,class_weight=class_weights)
        print("Training Complete! Saving model to \\models...")
        self.__savemodel()
        self.__set_users()
        self.__my_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', categories=self.__my_users)

    def __set_users(self):
        self.__movies = self.__movies.rename(columns={"movieid": "movieId"})
        full = pd.merge(self.__ratings, self.__movies, on='movieId')
        full.sort_values(by='rating')
        # NEW CODE
        data = [0] * 10
        for i in range(0, 10):
            index = i / 2. + 0.5
            data[i] = full[full['rating'] == index]

        df = pd.DataFrame()
        for i in range(0, 10):
            df = df.append(data[i].head(cfg.nn_head_size))

        # OLD CODE
        userids = df[['userId']].apply(np.array)
        tempenc = OneHotEncoder(sparse=False)
        tempenc.fit(userids)

        c = np.array(userids.userId.tolist())
        c = c.reshape(c.shape[0], 1)
        tempenc.fit(c)

        self.__my_users = tempenc.categories_


    def setupdata(self):
        self.__movies = self.__movies.rename(columns={"movieid": "movieId"})
        full = pd.merge(self.__ratings, self.__movies, on='movieId')
        full.sort_values(by='rating')
        full.genres = full.genres.str.split('|')
        # NEW CODE
        data = [0] * 10
        for i in range(0, 10):
            index = i / 2. + 0.5
            data[i] = full[full['rating'] == index]

        df = pd.DataFrame()
        for i in range(0, 10):
            df = df.append(data[i].head(cfg.nn_head_size))

        # OLD CODE
        size = len(df.userId.values)
        df.title = df.title.apply(self.__vectorize)

        df.genres = df.genres.apply(self.__myonehot)

        df[['userId', 'genres', 'rating']] = df[['userId', 'genres', 'rating']].apply(np.array)
        specific = df[['title', 'genres', 'userId', 'rating']].copy()
        target = specific.pop('rating')

        a = np.array(specific.title.tolist())
        b = np.array(specific.genres.tolist())
        c = np.array(specific.userId.tolist())

        c = c.reshape(c.shape[0], 1)

        c = self.__onehot(c)

        k = np.concatenate((a, b, c), axis=1)

        y = 2 * (target.values) - 1.

        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)
        dummy_y = to_categorical(encoded_y, 10)

        return size, k, dummy_y, y