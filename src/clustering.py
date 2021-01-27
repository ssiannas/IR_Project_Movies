from sklearn.cluster import KMeans
import pandas as pd

class Cluster():
    def __init__(self, Search):
        self.__search = Search
        self.__movies = self.__search.movies
        self.__ratings = self.__search.ratings
        self.__genres = self.__search.genres
        self.__clusters = int(input("How many clusters would you like (recommended: 8)\n"))
        self.__init_clustering()

    def __get_genre_ratings(self):
        genre_ratings = pd.DataFrame()
        for genre in self.__genres:
            genre_movies = self.__movies[self.__movies['genres'].str.contains(genre)]
            avg_genre_votes_per_user = \
                self.__ratings[self.__ratings['movieId'].isin(genre_movies['movieid'])].loc[:, ['userId', 'rating']].groupby(
                    ['userId'])[
                    'rating'].mean()

            genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)
        genre_ratings = genre_ratings.fillna(2.5)
        genre_ratings.columns = self.__genres
        genre_scores = genre_ratings.values.tolist()
        return  genre_scores

    def __init_clustering(self):
        data = self.__get_genre_ratings()
        clustering = KMeans(n_clusters=self.__clusters, random_state=5)
        clustering.fit(data)
        self.__my_clustering = clustering

    def set_user(self, userId):
        self.my_user_ratings = pd.DataFrame()
        for genre in self.__genres:
            genre_movies = self.__movies[self.__movies['genres'].str.contains(genre)]
            avg_genre_votes_per_user = \
                self.__ratings[self.__ratings['movieId'].isin(genre_movies['movieid'])].loc[
                    self.__ratings['userId'] == userId].groupby(
                    ['userId'])[
                    'rating'].mean()
            self.my_user_ratings = pd.concat([ self.my_user_ratings, avg_genre_votes_per_user], axis=1)
        self.my_user_ratings =  self.my_user_ratings.fillna(2.5)
        self.user_values = self.my_user_ratings.values

    def predict_user(self):
        self.user_values = self.my_user_ratings.values
        res = self.__my_clustering.predict(self.user_values)
        my_result = self.__my_clustering.cluster_centers_[res]
        return my_result[0]