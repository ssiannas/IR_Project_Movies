import src.config as cfg
from elasticsearch import Elasticsearch, helpers
from src.fromfile import csv_load
import pandas as pd
import math
from src.clustering import Cluster
from src.nn import NeuralNetwork

class Search():
    def __init__(self):
        self.__es = Elasticsearch([{'host': cfg.host, 'port': cfg.port}])
        self.__kmeans_enabled = False
        self.__nn_enabled = False
        self.movies = pd.read_csv('data\\movies.csv')
        self.ratings = pd.read_csv('data\\ratings.csv')
        self.__get_genres()
        self.__map_genres()

    def __str__(self):
        counter = 0
        for i in sorted(self.result.items(), key=lambda item: item[1], reverse=True):
            print(i)
            counter+=1
            if counter > cfg.print_no:
                break
        return ''

    def enable_kmeans(self):
        self.__kmeans_enabled = True
        self.__my_clustering = Cluster(self)

    def enable_nn(self):
        self.__nn_enabled = True
        self.__nn = NeuralNetwork(self)
        choice = int(input("Would you like to:\n"
              "1. Load neural network model from file\n"
              "2. Train the neural again.\n"))
        if choice == 1:
            self.__nn.loadmodel()
        else:
            size, xtrain, ytrain, ylabels = self.__nn.setupdata()
            self.__nn.train(xtrain,ytrain,ylabels,size)

    def query(self,title,*userid):
        if not len(userid):
            self.result = self.__search(title)
        else:
            userid = userid[0]
            self.result = self.__search_with_user(title, userid)

    def __search(self,title):
        res = self.__es.search(index="movieid", body={"query": {"match": {'title': title}}}, size=cfg.elasticsearch_result_cap, from_=0)
        hits = res['hits']['hits']
        num = res['hits']['total']['value']
        print("{} results found".format(num))
        for i in hits:
            print(i['_source']['title'])
        return

    def __search_with_user(self,title,userid):
        self.__myuser = self.ratings.loc[self.ratings['userId'] == userid]
        
        # query for the movie attributes
        body = {"query": {"match": {'title': title}}}
        res = helpers.scan(self.__es, query=body, index="movieid")
        hits = pd.DataFrame.from_dict([document['_source'] for document in res])
        # query for the elasticsearch scores
        res = self.__es.search(index="movieid", body=body, size=10000, from_=0)
        scores = [x["_score"] for x in res['hits']['hits']]
        hits.insert(len(hits.keys()), 'elastic_scores', scores)
        hits = self.get_table_full(hits)

        return self.__calculate_metric(hits, userid)


    def get_table_full(self, hits):
        meanscores = self.ratings.groupby('movieId')['rating'].mean().to_frame()

        # sort by movieid
        hits['sort'] = hits['movieid'].str.extract('(\d+)', expand=False).astype(int)
        hits.sort_values('sort', inplace=True)
        hits = hits.drop('sort', axis=1)

        # Keep only searched movies
        rows = hits['movieid'].tolist()
        rows = list(map(int, rows))

        boolean_series = meanscores.index.isin(rows)
        myscores = meanscores[boolean_series]

        boolean_series = self.__myuser.movieId.isin(rows)
        filtereduser = self.__myuser[boolean_series]

        # Create new ID columns
        with pd.option_context('mode.chained_assignment', None):
            hits['movieid'] = hits['movieid'].astype(int)
            hits['inds'] = hits.movieid.copy()
            filtereduser['inds'] = filtereduser.movieId.copy()

        filtereduser = filtereduser.set_index('inds')
        hits = hits.set_index('inds')

        # insert mean rating
        hits.insert(len(hits.keys()), 'mean_score', myscores['rating'].to_frame())

        # Append user ratings
        hits.insert(len(hits.keys()), 'user_score', filtereduser['rating'])
        return hits

    def __calculate_metric(self,hits, *user_id):

        my_user_id = user_id[0]
        cats = hits.genres.tolist()

        for index, cat in enumerate(cats):
            cats[index] = cat.split('|')
        final = {}
        # Cases for the application

        # Case 1: No Clustering OR Neural Network
        for tit, movie_categories, elastic_score, mean_score, user_score in zip(hits.title, cats, hits.elastic_scores, hits.mean_score, hits.user_score):
            score = elastic_score
            coef = 1
            if not math.isnan(mean_score):
                score += mean_score
                coef += 1
            if not math.isnan(user_score):
                score += user_score
                coef += 1
            else:
                if self.__kmeans_enabled:
                    user_means = self.__my_clustering.predict_user(my_user_id)
                    cat_count = 0
                    predicted_user_score = 0
                    for categ in movie_categories:
                        if categ != '(no genres listed)':
                            k = user_means[cfg.mapping[categ]]
                            predicted_user_score += k
                            cat_count += 1

                    if self.__nn_enabled:
                        predicted_user_score*=0.7
                        predicted_user_score += 0.3*self.__nn.predict(my_user_id,tit,movie_categories)

                    if predicted_user_score > 0:
                        predicted_user_score /= cat_count
                        score += predicted_user_score
                        coef += 1

                elif self.__nn_enabled:
                    predicted_nn_score = self.__nn.predict(my_user_id,tit,movie_categories)
                    score += predicted_nn_score
                    coef += 1


            final[tit] = [score / coef, coef]

        return final


    def __map_genres(self):
        count = 0
        # mapping
        for i in self.genres:
            cfg.mapping[i] = count
            count += 1

    def __get_genres(self):
        genres = []
        for i in self.movies.genres:
            i = i.split('|')
            for j in i:
                if j not in genres:
                    genres.append(j)
        self.all_genres = genres
        self.genres = genres[:-1]


    # ES Related functions
    def __load_es(self, filename):
        csv_load(self.__es, filename)

    def count_indices(self):
        res = self.__es.search(index="movieid", body={"query": {"match_all": {}}})
        print(res['hits']['total']['value'])

    def __empty_es(self, index_name):
        self.__es.indices.delete(index=index_name, ignore=[400, 404])

    def init_es(self):
        print("Initialising Elasticsearch")
        self.__empty_es('movieid')
        self.__load_es('movies')
        print("Initialization Complete!")


