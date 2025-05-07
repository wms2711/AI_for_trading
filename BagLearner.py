import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost, verbose=False):
        self.learners = [learner(**kwargs) for _ in range(bags)]
        self.verbose = verbose

    def author(self):
        return "mwang709"

    def study_group(self):
        return "mwang709"

    # Train each learner on a subset of the data
    def add_evidence(self, data_x, data_y):
        n = data_x.shape[0]
        for learn in self.learners:
            # Randomly select data points with replacement. this ensures that each learner sees a slightly different subset of the data.
            # points = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            points = np.random.randint(n, size=n)
            # Train leaner on the subset of data
            learn.add_evidence(data_x[points], data_y[points])

    # Predict label for given data points
    def query(self, points):
        # Traverse tree to find label for each data point
        array = []
        for learn in self.learners:
            array.append(learn.query(points))
        return array[0] if len(array) == 1 else self.most_common_element(np.vstack(array))

    def most_common_element(self, array):
        mce = np.zeros(array.shape[1])
        for index in range(array.shape[1]):
            # Start finding counts
            values, counts = np.unique(array[:, index], return_counts=True)
            mce[index] = values[np.argmax(counts)]
        return mce
