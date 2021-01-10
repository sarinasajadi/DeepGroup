'''
Created on Dec 1, 2020
Config File

@author: Sarina Sajadi
'''

class Config(object):
    def __init__(self):
        self.path = 'dataset/sushi'
        self.number_of_versions = 20
        self.num_users = 5000
        self.num_items = 10
        self.embedding_size_user = 64
        self.embedding_size_item = 64
        self.epoch = 100
        self.batch_size = 4096
        self.train_portion = 70
        self.hidden_units = [64, 32, 16, 8]
        self.k_participation = 0
        # self.k_participation = [1, 2, 3, 5, 10, 20]

        self.decision_rule = "plurality"
        #borda, plurality_borda

        self.similarity_param = [0, 0.25, 0.75]
        self.number_of_groups = [1000, 2000]
        self.max_len_groups = 10

