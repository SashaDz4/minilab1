# Description: Custom implementation of KNN algorithm
def calculate_index2distance(x_train, x_test) -> dict:
    idx2dist = {}
    for x_test_point in x_test:
        distances = []
        for i, x_train_point in enumerate(x_train):
            distance = ((x_train_point[0] - x_test_point[0]) ** 2 + (x_train_point[1] - x_test_point[1]) ** 2) ** 0.5
            distances.append((distance, i))
        idx2dist[tuple(x_test_point)] = distances
    return idx2dist


def get_current_class(neighbors: list[int]):
    counter = 0
    curr_class = None
    reversed_neighbor = neighbors[::-1]

    for c in reversed_neighbor:
        if neighbors.count(c) > counter:
            counter = neighbors.count(c)
            curr_class = c
    return curr_class


def predict(x_train, y_train, x_test, k) -> list[int]:
    """
    :param x_train: list of lists of floats
    :param y_train: list of ints
    :param x_test: list of lists of floats
    :param k: int
    :return: list of ints
    """
    idx2dist = calculate_index2distance(x_train, x_test)
    idx2dist = {k: sorted(v, key=lambda x: x[0]) for k, v in idx2dist.items()}

    neighbors = [[y_train[idx2dist[tuple(x_test_point)][i][1]] for i in range(k)] for x_test_point in x_test]
    clases = []
    for neighbor in neighbors:
        curr_class = get_current_class(neighbor[::-1])
        clases.append(curr_class)
        print(f"Point {x_test[neighbors.index(neighbor)]} has neighbors {neighbor} and class {curr_class}")

    return clases


if __name__ == '__main__':
    x_train_data = [[1, 2], [2, 1], [3, 3], [4, 4], [5, 5]]
    y_train_data = [1, 1, 2, 2, 3]
    x_test_data = [[3, 0], [4, 3], [5, 4]]
    k = 4

    predict(x_train_data, y_train_data, x_test_data, k)
