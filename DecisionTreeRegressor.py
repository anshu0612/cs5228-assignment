import numpy as np
import os
import json
import operator

class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=2):
        '''
        Initialization
        :param max_depth, type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split, type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # Calculate the error for a split dataset
    def calc_least_square_error(self, region1_y, region2_y):
        region1_error = np.sum(np.square(region1_y - np.mean(region1_y))) if region1_y.shape[0] > 0 else 0
        region2_error = np.sum(np.square(region2_y - np.mean(region2_y))) if region2_y.shape[0] > 0 else 0
        return region1_error + region2_error

    # Split the dataset based on the selected attribute threshold
    def split_data(self, attr_idx, value, x_train, y_train):
        left_x, right_x, left_y, right_y = list(), list(), list(), list()
        for i in range(x_train.shape[0]):
            if x_train[i][attr_idx] <= value:
                left_x.append(x_train[i])
                left_y.append(y_train[i])
            else:
                right_x.append(x_train[i])
                right_y.append(y_train[i])
        return np.array(left_x), np.array(right_x), np.array(left_y), np.array(right_y)

    # Get the split point with least error
    def get_split(self, x_train, y_train):
        splitting_variable, splitting_threshold = None, None
        least_error = np.inf
        out_l, out_r = 0, 0
        train_x_l, train_x_r = None, None
        train_y_l, train_y_r = None, None
        # iterate over each attribute
        for attr_idx in range(x_train.shape[1]):
            # iterate over each attribute value
            for row_idx in range(x_train.shape[0]):
                l_x, r_x, l_y, r_y = self.split_data(attr_idx, x_train[row_idx][attr_idx], x_train, y_train)
                error = self.calc_least_square_error(l_y, r_y)
                if error < least_error:
                    splitting_variable = attr_idx
                    splitting_threshold = x_train[row_idx][attr_idx]
                    least_error = error
                    out_l = np.mean(l_y) if l_y.shape[0] > 0 else 0
                    out_r = np.mean(r_y) if r_y.shape[0] > 0 else 0
                    train_x_l, train_x_r = l_x, r_x
                    train_y_l, train_y_r = l_y, r_y

        node_params = {
            "splitting_variable": splitting_variable,
            "splitting_threshold": splitting_threshold,
            "left": out_l,
            "right": out_r
         }

        splitted_set = {
            "left": {
                "train_x": train_x_l,
                "train_y": train_y_l
            },
            "right": {
                "train_x": train_x_r,
                "train_y": train_y_r
            }
        }

        return node_params, splitted_set

    # Recursively create nodes and split
    def split_node(self, node, splitted_data, depth):
        # check for max depth
        if depth > self.max_depth:
            return

        # Check left child
        left_child = splitted_data["left"]
        if left_child["train_x"].shape[0] >= self.min_samples_split:
            node["left"], new_split = self.get_split(left_child["train_x"], left_child["train_y"])
            self.split_node(node["left"], new_split, depth + 1)

        # Check right child
        right_child = splitted_data["right"]
        if right_child["train_x"].shape[0] >= self.min_samples_split:
            node["right"], new_split = self.get_split(right_child["train_x"], right_child["train_y"])
            self.split_node(node["right"], new_split,  depth + 1)

    # Build a Regression tree
    def build_regressor_tree(self, x_train, y_train):
        self.root, splitted_data = self.get_split(x_train, y_train)
        self.split_node(self.root, splitted_data, 0)
        return self.root

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''
        self.root = self.build_regressor_tree(X, y)

    def get_prediction(self, node, x_train):
        if x_train[node['splitting_variable']] <= node['splitting_threshold']:
            if isinstance(node['left'], dict):
                return self.get_prediction(node['left'], x_train)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.get_prediction(node['right'], x_train)
            else:
                return node['right']

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        y_pred = np.zeros((X.shape[0],))
        for i in range(X.shape[0]):
            y_pred[i] = self.get_prediction(self.root, X[i])
        return y_pred

    def get_model_dict(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


def compare_json_dic(json_dic, sample_json_dic):
    if isinstance(json_dic, dict):
        result = 1
        for key in sample_json_dic:
            if key in json_dic:
                result = result * compare_json_dic(json_dic[key], sample_json_dic[key])
                if result == 0:
                    return 0
            else:
                return 0
        return result
    else:
        rel_error = abs(json_dic - sample_json_dic) / np.maximum(1e-8, abs(sample_json_dic))
        if rel_error <= 1e-5:
            return 1
        else:
            return 0


def compare_predict_output(output, sample_output):
    rel_error = (abs(output - sample_output) / np.maximum(1e-8, abs(sample_output))).mean()
    if rel_error <= 1e-5:
        return 1
    else:
        return 0

# For test
if __name__=='__main__':
    for i in range(1):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split = j + 2)
            tree.fit(x_train, y_train)

            model_dict = tree.get_model_dict()
            y_pred = tree.predict(x_train)

            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_dict = json.load(fp)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
            if compare_json_dic(model_dict, test_model_dict) * compare_predict_output(y_pred, y_test_pred) == 1:
                print("True")
            else:
                print("False")



