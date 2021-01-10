import random
import pandas as pd
import itertools
from config import Config

def get_train_test_datasets(lambda_amount, num_groups, overlap_val, max_len_groups, num_items, fileName, version, train_portion=100):
    config = Config()
    file_name = fileName + '/negative_example_'+ config.decision_rule +'/ne_' + str(num_groups) + 'g-lambda' + str(lambda_amount) + 'version_' + str(version) + '.csv'
    if overlap_val > 0:
        file_name = fileName + '/overlap_negative_example_'+ config.decision_rule + '/ne_' + str(overlap_val) + 'overlap-g-lambda' + str(lambda_amount) + 'version_' + str(version) + '.csv'

    dataframe = pd.read_csv(file_name, error_bad_lines=False, sep='\t')

    train_number = int(train_portion * num_groups / 100)
    items = [i for i in range(num_items)]

    ratings = dataframe.rating.astype('category').cat.codes.values
    rankings = dataframe.ranking.astype('category').cat.codes.values

    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    ratings = list(divide_chunks(ratings, num_items))

    rankings = list(divide_chunks(rankings, num_items))
    test_rankings = rankings[train_number:]

    decisions = []
    for i in range(len(ratings)):
        for j in range(len(ratings[i])):
            if ratings[i][j] == 1:
                decisions.append(j)

    groups = []
    for index, row in dataframe.iterrows():
        value = []
        for i in range(max_len_groups):
            if 'user' in str(row[i + 1]):
                value.append(int(row[i + 1].replace('user', '')))
        groups.append(value)
    groups = list(k for k,_ in itertools.groupby(groups))

    x_train = [[], []]
    y_train = []
    x_test = [[], []]
    y_test = []
    for i in range(len(groups)):
        if i < train_number:
            for j in range(len(groups[i])):
                for k in range(num_items):
                    x_train[0].append(groups[i][j])
                    x_train[1].append(items[k])
                    if decisions[i] == k:
                        y_train.append(1)
                    else:
                        y_train.append(0)
        else:
            for j in range(len(groups[i])):
                for k in range(len(items)):
                    x_test[0].append(groups[i][j])
                    x_test[1].append(items[k])
                    if decisions[i] == k:
                        y_test.append(1)
                    else:
                        y_test.append(0)

    return groups, decisions, x_train, y_train, x_test, y_test, test_rankings

def get_personal_preferencs(fileName, number_of_users, n_items):
    if number_of_users == 0:
        dataframe = pd.read_csv('dataset/' + fileName + '_' + 'user_ranking.csv', sep='\t',
                                error_bad_lines=False)
    else:
        dataframe = pd.read_csv('dataset/' + fileName + '_' + str(number_of_users) + 'user_ranking.csv', sep='\t',
                                error_bad_lines=False)

    dict = {}
    for index, row in dataframe.iterrows():
        value = (
        row['item0'], row['item1'], row['item2'], row['item3'], row['item4'], row['item5'], row['item6'], row['item7'],
        row['item8'], row['item9'])

        dict['user' + str(index)] = value

    # make set of users
    users = []
    for key, value in dict.items():
        users.append(key)

    # making preferences
    prefs = []
    for i in range(len(users)):
        inner_prefs = []
        for j in range(n_items):
            if dict[users[i]][j] == 0:
                inner_prefs.append(1)
            else:
                inner_prefs.append(0)
        prefs.append(inner_prefs)
    return prefs

def get_train_test_datasets_reverse(lambda_amount, num_groups, overlap_val, max_len_groups, num_items, fileName, version, train_portion=100):
    file_name = fileName + '/negative_example_'+ Config.decision_rule +'/ne_' + str(num_groups) + 'g-lambda' + str(
        lambda_amount) + 'version_' + str(version) + '.csv'
    if overlap_val > 0:
        file_name = fileName + '/overlap_negative_example_'+Config.decision_rule +'/ne_' + str(
            overlap_val) + 'overlap-g-lambda' + str(
            lambda_amount) + 'version_' + str(version) + '.csv'

    dataframe = pd.read_csv(file_name, error_bad_lines=False, sep='\t')
    train_number = int(train_portion * num_groups / 100)
    items = [i for i in range(num_items)]

    ratings = dataframe.rating.astype('category').cat.codes.values
    rankings = dataframe.ranking.astype('category').cat.codes.values

    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    ratings = list(divide_chunks(ratings, num_items))
    # print(ratings)

    rankings = list(divide_chunks(rankings, num_items))
    test_rankings = rankings

    decisions = []
    for i in range(len(ratings)):
        for j in range(len(ratings[i])):
            if ratings[i][j] == 1:
                decisions.append(j)
    # print(decisions)

    groups = []
    for index, row in dataframe.iterrows():
        value = []
        for i in range(max_len_groups):
            if 'user' in str(row[i + 1]):
                value.append(int(row[i + 1].replace('user', '')))
        groups.append(value)
    groups = list(k for k,_ in itertools.groupby(groups))

    if overlap_val > 0:
        personal_prefs = get_personal_preferencs(fileName, Config.num_users, num_items)
    else:
        personal_prefs = get_personal_preferencs(fileName, 0, num_items)

    x_train = [[], []]
    y_train = []
    x_test = [[], []]
    y_test = []
    final_test_ranking = []
    for i in range(len(groups)):
        for j in range(len(groups[i])):
            for k in range(num_items):
                x_train[0].append(groups[i][j])
                x_train[1].append(items[k])
                if decisions[i] == k:
                    y_train.append(1)
                else:
                    y_train.append(0)
                if groups[i][j] not in x_test:
                    x_test[0].append(groups[i][j])
                    x_test[1].append(items[k])
                    y_test.append(personal_prefs[groups[i][j]][k])
                final_test_ranking.append(test_rankings[i])

    return groups, decisions, x_train, y_train, x_test, y_test, final_test_ranking


