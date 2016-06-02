import math
import time
import argparse


def get_prediction(mean_rating, actual_rating, weight):
    """
    Predict ratings for user and movie
    :param mean_rating:
    :param actual_rating:
    :param weight:
    :return:
    """

    numerator = weight * (actual_rating - mean_rating)
    denominator = abs(weight)
    return numerator, denominator


def pearson_correlation(user_x, user_y):
    """
    Computes pearson's correlation coefficient.
    :param user_x:
    :param user_y:
    :return:
    """
    n = len(user_x)
    if n == 0:
        return 0.0

    xbar = float(sum(user_x)) / n
    ybar = float(sum(user_y)) / n

    numer = sum([x*y for x,y in zip(user_x,user_y)]) - n*(xbar * ybar)
    denom = math.sqrt((sum([x*x for x in user_x]) - n * xbar**2)*(sum([y*y for y in user_y]) - n * ybar**2))

    if denom == 0:
        return 0.0

    return float(numer) / float(denom)


def get_similarity(user_i, user_j, user_map):
    """
    Function to calculate the weight/similarity between any pair of users
    :param user_i:
    :param user_j:
    :param user_map:
    :return:
    """

    movies_i = user_map[user_i]
    movies_j = user_map[user_j]
    common_movies = list((set(movies_i.keys())).intersection(set(movies_j.keys())))
    ratings1 = list()
    ratings2 = list()
    for movie_selected in common_movies:
        ratings1.append(movies_i[movie_selected])
        ratings2.append(movies_j[movie_selected])

    weight = pearson_correlation(ratings1, ratings2)

    return weight


def create_test_data_set(filename):
    """
    read the list of movie user and rating from test set
    :param filename:
    :return:
    """
    user_movie = list()
    try:
        with open(filename) as f:
            for line in f:
                items = line.split(",")
                movie_id = int(items[0].strip())
                user_id = int(items[1].strip())
                rating_test = float(items[2].strip())
                value = movie_id, user_id, rating_test
                user_movie.append(value)
        f.close()
    except Exception as e:
        print("({})".format(e))

    return user_movie


def create_data_set(filename):
    """
      Function to create a key value pair for movies and users from training data file,
      also returns average rating across all movies and users in training data
      :param filename:
      :return:
    """

    sum_rating = 0.0
    total_number_of_ratings = 0
    movie_user_dict = {}
    user_movie_rating_dict = {}

    try:
        with open(filename) as f:
            for line in f:
                users = list()
                movie_rating = {}
                items = line.split(",")
                movie_id = int(items[0].strip())
                user_id = int(items[1].strip())
                rating_current = float(items[2].strip())
                sum_rating += rating_current
                total_number_of_ratings+= 1
                # create movie_id-> user_id dictionary
                if movie_user_dict.has_key(movie_id):
                    users = movie_user_dict[movie_id]
                users.append(user_id)
                movie_user_dict[movie_id] = users

                # create user_id -> movie_id->rating, list average rating
                if user_movie_rating_dict.has_key(user_id):
                    movie_rating = user_movie_rating_dict[user_id]

                if not movie_rating.has_key(movie_id):
                    movie_rating[movie_id] = rating_current
                user_movie_rating_dict[user_id] = movie_rating
        f.close()
    except Exception as e:
        print("({})".format(e))

    if total_number_of_ratings > 0:
        avg_rating = sum_rating/float(total_number_of_ratings)

    return movie_user_dict, user_movie_rating_dict, avg_rating


def calculate_average_rating_per_user(user_id, user_movie_map):
    """
    Function to calculate average rating of movie for a user
    :param user_id:
    :param user_movie_map:
    :return:
    """
    avg_rating = 0.0
    movies_ratings = user_movie_map[user_id]
    n = len(movies_ratings)
    b_new_user = True
    if n > 0:
        avg_rating = float(sum(movies_ratings.values())) / n
        b_new_user = False

    return avg_rating, b_new_user


def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def collaborative_filtering(train_file, test_file):
    """
    Performs collaborative filtering using memory based technique
    :param train_file:
    :param test_file:
    :return:
    """
    movie_user, user_movie_rating, avg_rating_training = create_data_set(train_file)
    test_user_movie_rating = create_test_data_set(test_file)

    predicted_values_list = list()
    avg_rating_all_users = {}
    similarity_users = {}
    rmse_sum = 0.0
    mae_sum = 0.0

    for index, item in enumerate(test_user_movie_rating):
        movie = item[0]
        user = item[1]
        rating = item[2]
        is_new_movie = False
        is_new_user = False

        if user not in avg_rating_all_users:
            r_i_bar, is_new_user = calculate_average_rating_per_user(user, user_movie_rating)
            avg_rating_all_users[user] = r_i_bar
        else:
            r_i_bar = avg_rating_all_users[user]

        try:
            all_users_for_a_movie = movie_user[movie]
        except KeyError as ky:
            is_new_movie = True

            # check for new user
            if is_new_user:
                r_i_bar = avg_rating_training

        if not is_new_movie:
            predicted_value_num = 0.0
            predicted_value_den = 0.0

            for other in all_users_for_a_movie:
                r_j_k = user_movie_rating[other][movie]
                if other in avg_rating_all_users:
                    r_j_bar = avg_rating_all_users[other]
                else:
                    r_j_bar, is_new_user = calculate_average_rating_per_user(other, user_movie_rating)
                    avg_rating_all_users[other] = r_j_bar

                if similarity_users.has_key((user, other)):
                    weight_i_j = similarity_users[(user, other)]
                elif similarity_users.has_key((other, user)):
                    weight_i_j = similarity_users[(other, user)]
                else:
                    weight_i_j = get_similarity(user, other, user_movie_rating)
                    similarity_users[(user, other)] = weight_i_j
                predicted_value = get_prediction(r_j_bar, r_j_k, weight_i_j)
                predicted_value_num += predicted_value[0]
                predicted_value_den += predicted_value[1]

            try:
                predict_term = r_i_bar + (float(predicted_value_num) / float(predicted_value_den))
            except ZeroDivisionError as e:
                predict_term = r_i_bar

        elif is_new_movie:
            predict_term = r_i_bar

        predict_term = round(predict_term, 1)
        rmse_sum += (rating - predict_term)**2
        mae_sum += abs(rating - predict_term)
        val = movie, user, rating, predict_term
        predicted_values_list.append(val)


    number_of_test_items = len(predicted_values_list)

    rmse = math.sqrt(float(rmse_sum)/float(number_of_test_items))
    mae = float(mae_sum)/float(number_of_test_items)

    print "RootMeanSquareError: %.4f, MeanAbsoluteError: %.4f" %(rmse, mae)
    with open("predictions.txt", "w") as file_obj:
        for item in predicted_values_list:
            op_line_str = str(item[0]) + ',' + str(item[1]) + ',' + str(item[2]) + "," +str(item[3])
            file_obj.write(op_line_str+ "\n")
    file_obj.close()


if __name__ == '__main__':
    start = time.time()

    args = parse_argument()
    file_name_training = args['train'][0]
    file_name_test = args['test'][0]

    collaborative_filtering(file_name_training, file_name_test)
    # print "Process time: " + str(time.time() - start)



