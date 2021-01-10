'''
Created on Dec 1, 2020
Main function

@author: Sarina Sajadi
'''

import numpy as np
import pandas as pd
from generateGroups import get_train_test_datasets
from model import get_model
from config import Config

def mainFunction(config):
    if config.k_participation == 0:
        for simKey in range(len(config.similarity_param)):
            for groupKey in range(len(config.number_of_groups)):
                file_name = config.path + str(config.number_of_groups[groupKey]) + 'groups' + 'lambda' + str(config.similarity_param[simKey])
                test_acc = 0

                for versionKey in range(config.number_of_versions):
                    groups, decisions, x_train, y_train, x_test, y_test, test_rankings = get_train_test_datasets(
                        config.similarity_param[simKey], config.number_of_groups[groupKey], config.k_participation, config.max_len_groups,
                        config.num_items, config.path, versionKey, config.train_portion)

                    user_embeddings, item_embeddings = get_model(config.num_users, config.embedding_size_user, config.num_items,
                                                                 config.embedding_size_item,
                                                                 config.hidden_units, x_train, y_train, config.epoch,
                                                                 config.batch_size)
                    embedding_sum = 0
                    for i in range(len(user_embeddings)):
                        embedding_sum += user_embeddings[i]
                    average_embedding = embedding_sum / len(user_embeddings)

                    EPSILON = 1e-07

                    def cosine_similarities(x):
                        dot_pdts = np.dot(user_embeddings, x)
                        norms = np.linalg.norm(x) * np.linalg.norm(user_embeddings, axis=1)
                        return dot_pdts / (norms + EPSILON)

                    sim_list = cosine_similarities(average_embedding)
                    average_index = np.where(sim_list == np.amax(sim_list))[0][0]
                    average_user_index = np.where(sim_list == np.amax(sim_list))[0][0]

                    group_embedding = []
                    for i in range(len(groups)):
                        group_set = []
                        for j in range(len(groups[i])):
                            if groups[i][j] in x_train[0]:
                                group_set.append(user_embeddings[groups[i][j]])
                            else:
                                group_set.append(user_embeddings[average_user_index])
                        group_embedding.append(np.mean(group_set, axis=0))

                    train_number = int(config.train_portion * len(groups) / 100)

                    group_train = [[], []]
                    group_labels = []
                    group_test = [[], []]
                    group_labels_test = []
                    for i in range(len(groups)):
                        for j in range(config.num_items):
                            if i < train_number:
                                group_train[0].append(group_embedding[i])
                                group_train[1].append(item_embeddings[j])
                                if decisions[i] == j:
                                    group_labels.append(1)
                                else:
                                    group_labels.append(0)
                            else:
                                group_test[0].append(group_embedding[i])
                                group_test[1].append(item_embeddings[j])
                                if decisions[i] == j:
                                    group_labels_test.append(1)
                                else:
                                    group_labels_test.append(0)

                    y_test_pred = get_model(config.num_users, config.embedding_size_user, config.num_items,
                        config.embedding_size_item, config.hidden_units, group_train,
                        group_labels, config.epoch, config.batch_size, group_test, model_type='group')

                    def divide_chunks(l, n):
                        for i in range(0, len(l), n):
                            yield l[i:i + n]

                    # predicted_list[0] indicates the predictions for group 0 in test set over the n items
                    predicted_list = list(divide_chunks(y_test_pred, config.num_items))

                    actual_list = list(divide_chunks(group_labels_test, config.num_items))

                    test_acc_count = 0
                    for i in range(len(predicted_list)):
                        top_choice_index = np.argmax(predicted_list[i], axis=0)[0]
                        for j in range(config.num_items):
                            if actual_list[i][j] == 1 and j == top_choice_index:
                                test_acc_count += 1

                    # this is the accuracy of predicting the top choice correctly
                    test_acc += (test_acc_count * 100 / len(predicted_list))

                print('DeepGroup_test_acc', test_acc / config.number_of_versions)

                # initialize list of lists
                result_dataframe = pd.read_csv('dataset/model_results.csv', sep='\t', index_col=0)
                index = len(result_dataframe.index)
                data = [[file_name, round(test_acc / config.number_of_versions, 2)]]
                # Create the pandas DataFrame
                data_dataframe = pd.DataFrame(data, columns=['file_name', 'Test_accuracy'], index=[index])
                data_dataframe.to_csv("dataset/model_results.csv", mode='a', header=False)
    else:
        similarity_param = 0
        for kVal in range(len(config.k_participation)):
            file_name = config.path + str(config.k_participation[kVal]) + 'k-participation'
            test_acc = 0
            for versionKey in range(config.number_of_versions):
                fileData = pd.read_csv(config.path + '/overlap_negative_example_' + config.decision_rule + '/ne_' + str(
                    config.k_participation[kVal]) + 'overlap-g-lambda' + str(similarity_param) + 'version_' + str(versionKey) + '.csv')
                group_numbers = len(fileData.index) / config.num_items

                groups, decisions, x_train, y_train, x_test, y_test, test_rankings = get_train_test_datasets(
                    similarity_param, group_numbers, config.k_participation[kVal], config.max_len_groups,
                    config.num_items, config.path, versionKey, config.train_portion)

                user_embeddings, item_embeddings = get_model(config.num_users, config.embedding_size_user, config.num_items,
                                                             config.embedding_size_item, config.hidden_units, x_train, y_train, config.epoch, config.batch_size)
                embedding_sum = 0
                for i in range(len(user_embeddings)):
                    embedding_sum += user_embeddings[i]
                average_embedding = embedding_sum / len(user_embeddings)

                EPSILON = 1e-07

                def cosine_similarities(x):
                    dot_pdts = np.dot(user_embeddings, x)
                    norms = np.linalg.norm(x) * np.linalg.norm(user_embeddings, axis=1)
                    return dot_pdts / (norms + EPSILON)

                sim_list = cosine_similarities(average_embedding)
                average_index = np.where(sim_list == np.amax(sim_list))[0][0]
                average_user_index = np.where(sim_list == np.amax(sim_list))[0][0]

                group_embedding = []
                for i in range(len(groups)):
                    group_set = []
                    for j in range(len(groups[i])):
                        if groups[i][j] in x_train[0]:
                            group_set.append(user_embeddings[groups[i][j]])
                        else:
                            group_set.append(user_embeddings[average_user_index])
                    group_embedding.append(np.mean(group_set, axis=0))

                train_number = int(config.train_portion * len(groups) / 100)

                group_train = [[], []]
                group_labels = []
                group_test = [[], []]
                group_labels_test = []
                for i in range(len(groups)):
                    for j in range(config.num_items):
                        if i < train_number:
                            group_train[0].append(group_embedding[i])
                            group_train[1].append(item_embeddings[j])
                            if decisions[i] == j:
                                group_labels.append(1)
                            else:
                                group_labels.append(0)
                        else:
                            group_test[0].append(group_embedding[i])
                            group_test[1].append(item_embeddings[j])
                            if decisions[i] == j:
                                group_labels_test.append(1)
                            else:
                                group_labels_test.append(0)

                y_test_pred = get_model(config.num_users, config.embedding_size_user, config.num_items,
                          config.embedding_size_item, config.hidden_units, group_train, group_labels, config.epoch,
                          config.batch_size, group_test, model_type='group')

                def divide_chunks(l, n):
                    for i in range(0, len(l), n):
                        yield l[i:i + n]

                # predicted_list[0] indicates the predictions for group 0 in test set over the n items
                predicted_list = list(divide_chunks(y_test_pred, config.num_items))

                actual_list = list(divide_chunks(group_labels_test, config.num_items))

                test_acc_count = 0
                for i in range(len(predicted_list)):
                    top_choice_index = np.argmax(predicted_list[i], axis=0)[0]
                    for j in range(config.num_items):
                        if actual_list[i][j] == 1 and j == top_choice_index:
                            test_acc_count += 1

                # this is the accuracy of predicting the top choice correctly
                test_acc += (test_acc_count * 100 / len(predicted_list))

            print('DeepGroup_test_acc', test_acc / config.number_of_versions)

            # initialize list of lists
            result_dataframe = pd.read_csv('dataset/model_results.csv', sep='\t', index_col=0)
            index = len(result_dataframe.index)
            data = [[file_name, round(test_acc / config.number_of_versions, 2)]]
            # Create the pandas DataFrame
            data_dataframe = pd.DataFrame(data, columns=['file_name', 'Test_accuracy'], index=[index])
            data_dataframe.to_csv("dataset/model_results.csv", mode='a', header=False)

# initial parameter class
config = Config()
mainFunction(config)
