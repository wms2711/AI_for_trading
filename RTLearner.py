import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
        
    # Construct decision tree from data
    def add_evidence(self, data_x, data_y):
        self.tree = self._build_tree(data_x, data_y)
    
    # Build decision tree
    def _build_tree(self, data_x, data_y):
        # If data_x smaller than leaf size or labels are all the same, return the mean of data_y
        if data_x.shape[0] <= self.leaf_size or np.unique(data_y).size == 1:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        
        # Find best feature using correlations with label with random selection
        best_feature_index = np.random.randint(data_x.shape[1])

        # Split data based on best feature
        data_split = data_x[:, best_feature_index]
        split_value = np.median(data_split)

        # If values of the best feature are all the same, return the mean of data_y
        if np.unique(data_split).size == 1:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        
        # Recursively build left and right subtrees
        left_mask = data_split <= split_value
        right_mask = data_split > split_value
        if np.all(left_mask) or np.all(right_mask):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        left_tree = self._build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self._build_tree(data_x[right_mask], data_y[right_mask])
        
        # Combine root, left and right subtrees into a single tree
        return np.vstack(([best_feature_index, split_value, 1, left_tree.shape[0] + 1], left_tree, right_tree))

    # Predict label for given data points
    def query(self, points):
        # If tree is empty, return None
        if self.tree is None:
            return None
        
        # Traverse tree to find label for each data point
        array = []
        for point in points:
            node = 0

            # Continue traversing until reaching a leaf node
            while self.tree[node, 0] != -1:
                # Move to left child if point is less than split value
                if point[int(self.tree[node, 0])] <= self.tree[node, 1]:
                    node += int(self.tree[node, 2])
                # Move to right child if point is greater than split value
                else:
                    node += int(self.tree[node, 3])
            # Add predicted label to array
            array.append(self.tree[node, 1])
        return np.array(array)