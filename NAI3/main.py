import argparse
import json
import numpy as np


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score or recommend movies')
    parser.add_argument('--user1', dest='user1', required=True,
                        help='First user (or the target user for recommendations)')
    parser.add_argument('--user2', dest='user2',
                        help='Second user (required for similarity score)')
    parser.add_argument("--score-type", dest="score_type", required=True,
                        choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
    parser.add_argument('--recommend', dest='recommend', action='store_true',
                        help='Recommend movies for a specific user')
    return parser


def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    common_movies = {item for item in dataset[user1] if item in dataset[user2]}
    if len(common_movies) == 0:
        return 0

    squared_diff = [np.square(dataset[user1][item] - dataset[user2][item]) for item in common_movies]
    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    common_movies = {item for item in dataset[user1] if item in dataset[user2]}
    if len(common_movies) == 0:
        return 0

    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    num_ratings = len(common_movies)
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)


def recommend_movies(dataset, target_user, score_type, num_recommendations=5):
    if target_user not in dataset:
        raise TypeError('Cannot find ' + target_user + ' in the dataset')

    #zbior filmow jakie dana osoba juz widziala
    target_movies = set(dataset[target_user].keys())
    predicted_ratings = {}

    #ocena kazdej osoby po kolei wzgledem naszego usera
    for other_user in dataset:
        if other_user == target_user:
            continue

        if score_type == 'Euclidean':
            similarity = euclidean_score(dataset, target_user, other_user)
        elif score_type == 'Pearson':
            similarity = pearson_score(dataset, target_user, other_user)
        else:
            continue

        if similarity <= 0:
            continue

        for movie, rating in dataset[other_user].items():
            if movie not in target_movies:
                if movie not in predicted_ratings:
                    predicted_ratings[movie] = {'total_score': 0, 'similarity_sum': 0}
                predicted_ratings[movie]['total_score'] += similarity * rating #wynik to iloczyn tego jak uzytkownik jest do nas podobny i jaka ocene wystawil
                predicted_ratings[movie]['similarity_sum'] += similarity

    for movie in predicted_ratings:
        predicted_ratings[movie] = predicted_ratings[movie]['total_score'] / predicted_ratings[movie]['similarity_sum']

    recommended_movies = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)

    unrecommended_movies = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=False)

    return recommended_movies[:num_recommendations], unrecommended_movies[:num_recommendations]


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user1 = args.user1
    score_type = args.score_type
    ratings_file = 'data.json'

    with open(ratings_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    if args.recommend:
        print(f"Recommended movies for: {user1}")
        recommended_movies, unrecommended_movies = recommend_movies(data, user1, score_type)

        print("\nTop 5 recommended movies:")
        for movie, predicted_rating in recommended_movies[:5]:
            print(f"{movie}: {predicted_rating:.2f}")

        print("\nTop 5 unrecommended movies:")
        for movie, predicted_rating in unrecommended_movies[:5]:
            print(f"{movie}: {predicted_rating:.2f}")

    else:
        print(f"Wrong score_type")
