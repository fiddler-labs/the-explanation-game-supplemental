# Adapted from https://github.com/Behrouz-Babaki/COP-Kmeans (MIT License)

import numpy as np
import scipy.spatial.distance


def proximity_kmeans(dataset, scores, n_clusters, max_distance=None,  # noqa C901
                     initialization='kmpp',
                     max_iter=300, tol=1e-4, random_state=0):
    """A special case of COP n_clusters-means where each point has a score, and
    the constraints are such that points are constrained to be in groups only
    with points whose scores differ by max_distance or less.

    If max_distance=None, we run unconstrained n_clusters-means.
    """
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    tol = tolerance(tol, dataset)

    print(f'Initializing centers via method "{initialization}"...')
    centers = initialize_centers(dataset, n_clusters, initialization,
                                 random_state=random_state)

    print('Running proximity kmeans...')
    for _ in range(max_iter):
        clusters_ = np.full(dataset.shape[0], np.nan)
        min_scores = np.full(n_clusters, np.nan)
        max_scores = np.full(n_clusters, np.nan)
        for i, (d, score) in enumerate(zip(dataset, scores)):
            cluster_ids_by_distance, _ = closest_clusters(centers, d)
            if np.isnan(clusters_[i]):
                for next_nearest_cluster_id in cluster_ids_by_distance:
                    # check for constraint violation
                    if max_distance is None:
                        violate_constraints = False
                    else:
                        if not np.isnan(min_scores[next_nearest_cluster_id]):
                            distance_from_min = abs(
                                score - min_scores[next_nearest_cluster_id])
                        else:
                            distance_from_min = 0

                        if not np.isnan(max_scores[next_nearest_cluster_id]):
                            distance_from_max = abs(
                                score - max_scores[next_nearest_cluster_id])
                        else:
                            distance_from_max = 0
                        violate_constraints = (
                                distance_from_min > max_distance
                                or distance_from_max > max_distance)

                    # assign point to cluster if allowed by constraints
                    if not violate_constraints:
                        # assign point to cluster
                        clusters_[i] = next_nearest_cluster_id

                        # update cluster min/max
                        if (np.isnan(min_scores[next_nearest_cluster_id])
                                or score < min_scores[
                                    next_nearest_cluster_id]):
                            min_scores[next_nearest_cluster_id] = score
                        if (np.isnan(max_scores[next_nearest_cluster_id])
                                or score > max_scores[
                                    next_nearest_cluster_id]):
                            max_scores[next_nearest_cluster_id] = score

                        break

                # if we iterate through the entire for loop without breaking,
                # this means we failed to find a suitable cluster
                else:
                    raise Exception('Cannot satisfy constraints, try '
                                    'increasing n_clusters or max_distance')

        clusters_, centers_ = compute_centers(clusters_, dataset, n_clusters,
                                              random_state=random_state)
        shift = sum(l2_distance(centers[i], centers_[i])
                    for i in range(n_clusters))
        if shift <= tol:
            break

        centers = centers_

    return clusters_, centers_


def l2_distance(point1, point2):
    return scipy.spatial.distance.euclidean(point1, point2)


# taken from scikit-learn (https://github.com/scikit-learn/scikit-learn/blob/919b4a8fbdf0e313fb702bd083abb31d67f7d8f9/sklearn/cluster/k_means_.py#L160)  # noqa E501
def tolerance(tol, dataset):
    n = len(dataset)
    dim = len(dataset[0])
    averages = [sum(dataset[i][d] for i in range(n)) / float(n) for d in
                range(dim)]
    variances = [
        sum((dataset[i][d] - averages[d]) ** 2 for i in range(n)) / float(n)
        for d in range(dim)]
    return tol * sum(variances) / dim


def closest_clusters(centers, datapoint):
    distances = [l2_distance(center, datapoint) for
                 center in centers]
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances


def initialize_centers(dataset, k, method, random_state=None):
    if method == 'random':
        if random_state is None:
            ids = np.random.permutation(dataset.shape[0])
        else:
            ids = random_state.permutation(dataset.shape[0])
        return dataset[ids[:k], :]

    elif method == 'kmpp':
        chances = [1] * len(dataset)
        centers = []

        for _ in range(k):
            chances = [x / sum(chances) for x in chances]
            if random_state is None:
                r = np.random.rand()
            else:
                r = random_state.rand()
            acc = 0.0
            for index, chance in enumerate(chances):
                if acc + chance >= r:
                    break
                acc += chance
            centers.append(dataset[index])

            for index, point in enumerate(dataset):
                cids, distances = closest_clusters(centers, point)
                chances[index] = distances[cids[0]]

        return np.array(centers)


def compute_centers(clusters, dataset, k, random_state=None):
    # re-number clusters to push empty clusters to the end
    np.unique(clusters)
    populated_clusters = np.unique(clusters)
    cluster_id_map = {cluster_id: i for i, cluster_id in
                      enumerate(populated_clusters)}
    for i in range(clusters.shape[0]):
        clusters[i] = cluster_id_map[clusters[i]]

    # compute centers of populated clusters
    centers = np.zeros((k, dataset.shape[1]))
    for cluster_id in range(populated_clusters.shape[0]):
        centers[cluster_id, :] = dataset[clusters == cluster_id].mean(axis=0)

    # re-assign centers of empty clusters randomly
    if populated_clusters.shape[0] < k:
        for cluster_id in range(populated_clusters.shape[0], k):
            if random_state is not None:
                if isinstance(random_state, int):
                    random_state = np.random.RandomState(random_state)
                row_id = random_state.choice(dataset.shape[0])
            else:
                row_id = np.random.choice(dataset.shape[0])
            centers[cluster_id] = dataset[row_id]

    return clusters, centers
