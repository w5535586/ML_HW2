import numpy as np
import sklearn 
from scipy import spatial
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.metrics import accuracy_score
from collections import Counter


###############################  LinearClassifier  ###############################
class KernelLinearClassifier:
    def __init__(self, learning_rate=0.01, num_iterations=100, gamma=1.0):
        """
        初始化線性分類器的屬性

        Parameters:
        - learning_rate (float): 學習速率，預設為0.01。
        - num_iterations (int): 訓練迭代次數，預設為100。
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.loss_history = []
        self.val_loss_history = []
        self.gamma = gamma

    def fit(self, X_train, y_train, X_val, y_val, re=True):
        X_val = self.rbf_kernel(X_val, X_train)
        self.X_train = X_train
        X_train = self.rbf_kernel(X_train, X_train)
        # 獲取訓練數據的樣本數和特徵數
        num_samples, num_features = X_train.shape
        
        # 初始化權重和偏差
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 開始訓練迭代
        for iter in range(self.num_iterations):
            # 計算線性模型的預測值
            linear_model = np.dot(X_train, self.weights) + self.bias
            # 使用sigmoid函數得到二元分類的預測值
            y_pred = self.sigmoid(linear_model)

            # 將預測值轉換為二元分類的類別預測
            y_cls_pred = [1 if i > 0.5 else 0 for i in y_pred]

            # 計算訓練準確度
            train_accuracy = sklearn.metrics.accuracy_score(y_train, y_cls_pred)
            
            # 如果需要，顯示訓練準確度
            if re:
                print("第{}次訓練".format(iter+1))
                print("Training Accuracy:", train_accuracy)

            # 計算梯度
            dw = (1 / num_samples) * np.dot(X_train.T, (y_pred - y_train))
            db = (1 / num_samples) * np.sum(y_pred - y_train)

            # 更新權重和偏差
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 計算訓練損失並記錄歷史
            train_loss = self.compute_loss(y_train, y_pred)
            self.loss_history.append(train_loss)
            
            # 計算驗證集上的線性模型和損失
            val_linear_model = np.dot(X_val, self.weights) + self.bias
            val_y_pred = self.sigmoid(val_linear_model)
            val_loss = self.compute_loss(y_val, val_y_pred)

            # 將驗證損失記錄到歷史中
            self.val_loss_history.append(val_loss)

    def rbf_kernel(self, X1, X2):
        # Calculate the RBF kernel matrix
        pairwise_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * pairwise_dists)

    def predict(self, X):
        X = self.rbf_kernel(X, self.X_train)
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def compute_loss(self, y, y_pred):
        epsilon = 1e-15  # 防止log(0)
        loss = - (y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        return np.mean(loss)

    def plot_loss(self, save_path=None):
        plt.plot(range(1, self.num_iterations + 1), self.loss_history, label="Train Loss")
        plt.plot(range(1, self.num_iterations + 1), self.val_loss_history, label="Validation Loss")
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('LinearClassifier Loss over Iterations')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_feature_importance(self):
        """
        獲取特徵重要性

        Returns:
        - numpy.ndarray: 特徵重要性值的絕對值。
        """
        return np.abs(self.weights)
    
    def k_fold_cross_validation(self, X, y, k=5):
        # 初始化空的訓練和驗證精確度列表
        train_accuracies = []
        val_accuracies = []

        # 計算每個折疊的大小
        fold_size = len(X) // k

        # 將資料分成 k 個折疊
        folds_X = [X[i * fold_size:(i + 1) * fold_size] for i in range(k)]
        folds_y = [y[i * fold_size:(i + 1) * fold_size] for i in range(k)]
        
        # 創建一個模型列表來存儲每個折疊的模型
        model_list = []

        # 開始 K-Fold 交叉驗證的迴圈
        for i in range(k):
            # 將訓練數據和標籤合併，排除當前折疊
            X_train = np.concatenate([folds_X[j] for j in range(k) if j != i])
            y_train = np.concatenate([folds_y[j] for j in range(k) if j != i])

            # 獲取當前折疊的驗證數據和標籤
            X_val = folds_X[i]
            y_val = folds_y[i]

            # 創建一個線性分類器模型
            model = LinearClassifier()

            # 使用訓練數據進行模型擬合
            model.fit(X_train, y_train, X_val, y_val, re=False)

            # 計算訓練和驗證精確度
            train_accuracy = sklearn.metrics.accuracy_score(y_train, model.predict(X_train))
            val_accuracy = sklearn.metrics.accuracy_score(y_val, model.predict(X_val))

            # 將精確度添加到列表中
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # 顯示訓練和驗證精確度
            print("LinearClassifier訓練精確度: {:.2f}% | 驗證精確度: {:.2f}%".format(train_accuracy * 100, val_accuracy * 100))
            
            # 將當前折疊的模型添加到模型列表中
            model_list.append(model)
        
        # 顯示 K-Fold 交叉驗證完成的訊息
        print("LinearClassifier K-Fold 交叉驗證完成。")
        
        # 顯示平均訓練和驗證精確度
        print("LinearClassifier平均訓練精確度: {:.2f}%".format(np.mean(train_accuracies) * 100))
        print("LinearClassifier平均驗證精確度: {:.2f}%".format(np.mean(val_accuracies) * 100))
        
        # 返回模型列表
        return model_list
###############################  LinearClassifier  ###############################

###############################  KNN  ###############################
#https://devpress.csdn.net/python/62f62634c6770329307fc080.html
#https://zhuanlan.zhihu.com/p/23966698
class KdNode(object):
    def __init__(self, dom_elt, split, left, right, idx):
        self.dom_elt = dom_elt
        self.split = split
        self.left = left
        self.right = right
        self.idx = idx

class KdTree(object):
    def __init__(self, data, metric='euclidean'):
        self.k = data.shape[1]  # 資料的維度
        self.distance_metric = metric  # 計算距離的方法，預設為歐氏距離
        indices = np.arange(data.shape[0])
        self.root = self.createNode(data, indices, 0)

    def createNode(self, data, indices, split):
        if data.shape[0] == 0:  # 若沒有資料，回傳空節點
            return None
        splitPos = data.shape[0] // 2
        dataSort = data[data[:, split].argsort()]  # 根據分割軸的值排序資料
        indicesSort = indices[data[:, split].argsort()]  # 排序資料對應的索引
        medianNode = dataSort[splitPos]  # 中位數節點
        medianIdx = indicesSort[splitPos]  # 中位數節點對應的索引
        splitNext = (split + 1) % self.k  # 循環切換分割軸
        return KdNode(medianNode, split, self.createNode(dataSort[:splitPos], indicesSort[:splitPos], splitNext), self.createNode(dataSort[splitPos + 1:], indicesSort[splitPos + 1:], splitNext), medianIdx)

    def calculate_distance(self, point1, point2):
        if self.distance_metric == 'euclidean':  # 歐氏距離
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.distance_metric == 'manhattan':  # 曼哈頓距離
            return np.sum(np.abs(point1 - point2))
        elif self.distance_metric == 'minkowski':  # 三角不等式距離（Minkowski距離，此處p=3）
            p = 3
            return np.power(np.sum(np.power(np.abs(point1 - point2), p), 1/p))
        else:
            raise ValueError("無效的距離計算方式")

    def search_nearest(self, point, k=3):
        self.result = namedtuple('nearestInf', 'indices distances')
        self.nearestIndices = []  # 最近鄰的索引
        self.nearestDistances = []  # 最近鄰的距離
        max_distance = float('inf')  # 最大距離初始化為正無窮大

        def travel(node, depth):
            nonlocal max_distance

            if node is not None:
                axis = node.split
                if point[axis] < node.dom_elt[axis]:  # 根據分割軸選擇遍歷的子節點順序
                    next_node, other_node = node.left, node.right
                else:
                    next_node, other_node = node.right, node.left

                travel(next_node, depth + 1)  # 遞迴遍歷靠近目標點的子樹
                distance = self.calculate_distance(point, node.dom_elt)  # 計算目標點與當前節點的距離
                if len(self.nearestIndices) < k:
                    self.nearestIndices.append(node.idx)  # 若最近鄰列表不滿，直接加入
                    self.nearestDistances.append(distance)
                    max_distance = max(self.nearestDistances) if self.nearestDistances else float('inf')
                elif distance < max_distance:
                    max_index = self.nearestDistances.index(max_distance)
                    self.nearestIndices[max_index] = node.idx  # 若有更近的節點，替換最遠的最近鄰
                    self.nearestDistances[max_index] = distance
                    max_distance = max(self.nearestDistances)

                if abs(point[axis] - node.dom_elt[axis]) < max_distance or len(self.nearestIndices) < k:
                    travel(other_node, depth + 1)  # 若距離範圍內或最近鄰列表不滿，遍歷另一子樹

        travel(self.root, 0)  # 從樹根開始遞迴遍歷
        return np.array(self.nearestIndices)  # 回傳最近鄰的索引
    
class kNN():
    def __init__(self, k=3, metric='euclidean', p=None, gamma=1.0):
        self.k = k
        self.metric = metric
        self.p = p
        self.tree = None
        self.loss_history = []
        self.gamma = gamma

    def fit(self, x_train, y_train):
        self.X_train = x_train
        x_train = self.rbf_kernel(x_train, x_train)
        self.x_train = x_train
        self.y_train = y_train
        
        if self.metric == 'euclidean':
            #self.tree = spatial.cKDTree(self.x_train)
            self.tree = KdTree(self.x_train)
        elif self.metric == 'manhattan':
            #self.tree = spatial.cKDTree(self.x_train, metric='manhattan')
            self.tree = KdTree(self.x_train, metric='manhattan')
        elif self.metric == 'minkowski':
            #self.tree = spatial.cKDTree(self.x_train, p=self.p)
            self.tree = KdTree(self.x_train, metric='minkowski')
        else:
            raise ValueError('Supported metrics are euclidean, manhattan, and minkowski')
    
    def rbf_kernel(self, X1, X2):
        # Calculate the RBF kernel matrix
        pairwise_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * pairwise_dists)
    
    def predict(self, x_test):
        preds = []
        x_test = self.rbf_kernel(x_test,self.X_train)
        for i, test_row in enumerate(x_test):
            nearest_indices = self.tree.search_nearest(test_row, self.k)
            #nearest_indices = self.tree.query(test_row, self.k, distance_upper_bound=np.inf)[1]
            nearest_neighbours = self.y_train[nearest_indices[nearest_indices < len(self.x_train)]]
            majority = np.argmax(np.bincount(nearest_neighbours))
            preds.append(majority)
        return np.array(preds)
    
    def predict_loss(self, x_test, k):
        preds = []
        for test_row in x_test:
            nearest_indices = self.tree.search_nearest(test_row, k)
            #nearest_indices = self.tree.query(test_row, k, distance_upper_bound=np.inf)[1]
            nearest_neighbours = self.y_train[nearest_indices[nearest_indices < len(self.x_train)]]
            majority = np.argmax(np.bincount(nearest_neighbours))
            preds.append(majority)
        return np.array(preds)
    
    def custom_loss(self, x_test, y_test, k_list, save_path="KNN_plot.png"):
        # 这是一个示例自定义损失函数，你可以根据你的需求进行修改
        for k in k_list:
            if k <= 1:
                continue
            predictions = self.predict_loss(x_test, k)
            loss = 1-sklearn.metrics.accuracy_score(y_test, predictions)
            self.loss_history.append(loss)

        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('LinearClassifier Loss over Iterations')
        if save_path:
            plt.savefig(save_path)  # 保存图像到指定路径
        else:
            plt.show()

    def kfold_cross_validation(self, X, y, k=5):
        # 将数据分成K个折叠
        fold_size = len(X) // k
        folds_X = [X[i * fold_size:(i + 1) * fold_size] for i in range(k)]
        folds_y = [y[i * fold_size:(i + 1) * fold_size] for i in range(k)]

        accuracies = []
        model_list = []
        for i in range(k):
            # 选择第i个折叠作为验证集，其余作为训练集
            X_train = np.concatenate([folds_X[j] for j in range(k) if j != i])
            y_train = np.concatenate([folds_y[j] for j in range(k) if j != i])
            X_val = folds_X[i]
            y_val = folds_y[i]

            model = kNN()
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            accuracy = np.mean(predictions == y_val)
            print(f"kNN of Accuracy for Fold {i + 1}: {accuracy}")
            accuracies.append(accuracy)
            model_list.append(model)

        mean_accuracy = np.mean(accuracies)
        print(f'kNN of Mean Accuracy across {k} folds: {mean_accuracy}')
        return model_list
            
###############################  KNN  ###############################

###############################  Naïve Decision Tree Classifier  ###############################
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.feature_subset = None

    def fit(self, X, y, depth=0,  features_subset=None):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            self.label = np.argmax(np.bincount(y))
        else:
            if features_subset is None:
                if self.max_features is not None:
                    self.feature_subset = np.random.choice(X.shape[1], self.max_features, replace=False)
                else:
                    self.feature_subset = np.arange(X.shape[1])
            X_subset = X[:, self.feature_subset]
            self.split_feature, self.split_threshold = self.find_best_split(X_subset, y)
            if self.split_feature is not None:
                left_mask = X[:, self.split_feature] <= self.split_threshold
                right_mask = X[:, self.split_feature] > self.split_threshold
                if np.any(left_mask) and np.any(~left_mask):
                    self.left = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features)
                    self.right = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features)
                    self.left.fit(X[left_mask], y[left_mask], depth + 1)
                    self.right.fit(X[right_mask], y[right_mask], depth + 1)
                else:
                    if np.any(left_mask):
                        self.label = np.argmax(np.bincount(y[left_mask]))
                    else:
                        self.label = np.argmax(np.bincount(y[~left_mask]))
            else:
                self.label = np.argmax(np.bincount(y))

    def find_best_split(self, X, y):
        num_features = X.shape[1]
        best_split_feature = None
        best_split_threshold = None
        best_gini = float('inf')
        
        threshold_value = 0.01
        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])

            if len(unique_values) == 2: 
                thresholds = [unique_values.min(), (unique_values.min() + unique_values.max()) / 2, unique_values.max()]
            elif len(unique_values) <= 10:  # Categorical feature with <= 10 categories
                thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]
            else:
                # For features with more than 10 unique values, use percentiles
                thresholds = np.percentile(unique_values, np.linspace(0, 100, 10))
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                gini_left = self.calculate_gini(y[left_mask])
                gini_right = self.calculate_gini(y[right_mask])

                weighted_gini = (len(y[left_mask]) / len(y)) * gini_left + (len(y[right_mask]) / len(y)) * gini_right
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split_feature = feature
                    best_split_threshold = threshold
                    # Early stopping if the Gini impurity reduction is below a threshold
                    if best_gini < threshold_value:
                        return best_split_feature, best_split_threshold
        
        return best_split_feature, best_split_threshold

    def calculate_gini(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)
    

    def feature_importance(self, X, y):
        if not hasattr(self, 'split_feature'):
            return np.zeros(X.shape[1])  # Return all zeros if tree is not trained yet

        feature_importance = np.zeros(X.shape[1])
        self._accumulate_feature_importance(X, y, feature_importance)
        return feature_importance

    def _accumulate_feature_importance(self, X, y, feature_importance):
        if hasattr(self, 'split_feature'):
            feature_importance[self.split_feature] += self.calculate_gini(y)
            if hasattr(self, 'left'):
                self.left._accumulate_feature_importance(X, y, feature_importance)
                self.right._accumulate_feature_importance(X, y, feature_importance)

    def predict(self, X):
        if hasattr(self, 'split_feature') and self.split_feature is not None:
            is_left = X[:, self.split_feature] <= self.split_threshold
            y_pred = np.zeros(X.shape[0])
            if hasattr(self, 'left'):
                y_pred[is_left] = self.left.predict(X[is_left])
                y_pred[~is_left] = self.right.predict(X[~is_left])
            else:
                y_pred[is_left] = self.label
                y_pred[~is_left] = self.label
            return y_pred
        else:
            return np.full(X.shape[0], self.label)
        
        
    def k_fold_cross_validation(self, X, y, k=5, max_depth=None, max_features=None):
        # 将数据分成K个折叠
        fold_size = len(X) // k
        folds_X = [X[i * fold_size:(i + 1) * fold_size] for i in range(k)]
        folds_y = [y[i * fold_size:(i + 1) * fold_size] for i in range(k)]

        accuracies = []
        model_list = []
        for i in range(k):
            # 选择第i个折叠作为验证集，其余作为训练集
            X_train = np.concatenate([folds_X[j] for j in range(k) if j != i])
            y_train = np.concatenate([folds_y[j] for j in range(k) if j != i])
            X_val = folds_X[i]
            y_val = folds_y[i]

            # 初始化并训练决策树模型
            model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
            model.fit(X_train, y_train)

            # 预测验证集并计算准确度
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            accuracies.append(accuracy)
            model_list.append(model)

            print(f'DecisionTree of Fold {i + 1}: Accuracy = {accuracy}')

        # 计算平均准确度
        avg_accuracy = np.mean(accuracies)
        print(f'DecisionTree of Average Accuracy: {avg_accuracy}')
        return model_list

###############################  Naïve Decision Tree Classifier  ###############################

###############################  Prune Decision Tree Classifier  ###############################
class PruneDecisionTreeClassifier:
    def __init__(self, max_depth=5, max_features=None, min_samples_split=2):
        # 初始化函數，設定模型的參數
        self.max_depth = max_depth  # 樹的最大深度
        self.max_features = max_features  # 每個節點分割時使用的特徵數
        self.min_samples_split = min_samples_split  # 最小樣本數，小於該數不再分割節點
        self.feature_subset = None  # 存儲特徵子集

    def fit(self, X, y, depth=0,  features_subset=None):
        # 擬合訓練數據，建立決策樹
        if depth == self.max_depth or len(np.unique(y)) == 1:
            # 若滿足停止條件（最大深度或同一類別），設定節點的標籤為多數類別
            self.label = np.argmax(np.bincount(y))
        elif len(y) < self.min_samples_split:
            # 若滿足停止條件（樣本數小於最小分割樣本數），設定節點的標籤為多數類別
            self.label = np.argmax(np.bincount(y))
        else:
            if features_subset is None:
                if self.max_features is not None:
                    # 隨機選擇特徵子集，控制每個節點的特徵數
                    self.feature_subset = np.random.choice(X.shape[1], self.max_features, replace=False)
                else:
                    self.feature_subset = np.arange(X.shape[1])
            X_subset = X[:, self.feature_subset]
            self.split_feature, self.split_threshold = self.find_best_split(X_subset, y)
            if self.split_feature is not None:
                # 若找到最佳分割特徵和閾值，創建左右子樹
                left_mask = X[:, self.split_feature] <= self.split_threshold
                right_mask = X[:, self.split_feature] > self.split_threshold
                if np.any(left_mask) and np.any(~left_mask):
                    self.left = PruneDecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features, min_samples_split=self.min_samples_split)
                    self.right = PruneDecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features, min_samples_split=self.min_samples_split)
                    self.left.fit(X[left_mask], y[left_mask], depth + 1)
                    self.right.fit(X[right_mask], y[right_mask], depth + 1)
                else:
                    if np.any(left_mask):
                        self.label = np.argmax(np.bincount(y[left_mask]))
                    else:
                        self.label = np.argmax(np.bincount(y[~left_mask]))
            else:
                self.label = np.argmax(np.bincount(y))

    def find_best_split(self, X, y):
        # 找尋最佳分割特徵和閾值
        num_features = X.shape[1]
        best_split_feature = None
        best_split_threshold = None
        best_gini = float('inf')
        
        threshold_value = 0.01  # 停止條件：基尼不純度減少閾值
        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])

            if len(unique_values) == 2:  # 二元特徵
                thresholds = [unique_values.min(), (unique_values.min() + unique_values.max()) / 2, unique_values.max()]
            elif len(unique_values) <= 10:  # 小於等於10個類別的分類特徵
                thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]
            else:
                # 針對具有大於10個唯一值的特徵，使用百分位數
                thresholds = np.percentile(unique_values, np.linspace(0, 100, 10))
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                gini_left = self.calculate_gini(y[left_mask])
                gini_right = self.calculate_gini(y[right_mask])

                weighted_gini = (len(y[left_mask]) / len(y)) * gini_left + (len(y[right_mask]) / len(y)) * gini_right
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split_feature = feature
                    best_split_threshold = threshold
                    if best_gini < threshold_value:
                        # 若Gini不純度減少低於閾值，提前停止
                        return best_split_feature, best_split_threshold
        
        return best_split_feature, best_split_threshold

    def calculate_gini(self, y):
        # 計算基尼不純度
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)

    def feature_importance(self, X, y):
        # 計算特徵重要性
        if not hasattr(self, 'split_feature'):
            return np.zeros(X.shape[1])  # 如果樹尚未訓練，返回全為零的特徵重要性

        feature_importance = np.zeros(X.shape[1])
        self._accumulate_feature_importance(X, y, feature_importance)
        return feature_importance

    def _accumulate_feature_importance(self, X, y, feature_importance):
        if hasattr(self, 'split_feature'):
            feature_importance[self.split_feature] += self.calculate_gini(y)
            if hasattr(self, 'left'):
                self.left._accumulate_feature_importance(X, y, feature_importance)
                self.right._accumulate_feature_importance(X, y, feature_importance)

    def predict(self, X):
        # 進行預測
        if hasattr(self, 'split_feature') and self.split_feature is not None:
            is_left = X[:, self.split_feature] <= self.split_threshold
            y_pred = np.zeros(X.shape[0])
            if hasattr(self, 'left'):
                y_pred[is_left] = self.left.predict(X[is_left])
                y_pred[~is_left] = self.right.predict(X[~is_left])
            else:
                y_pred[is_left] = self.label
                y_pred[~is_left] = self.label
            return y_pred
        else:
            return np.full(X.shape[0], self.label)

    def k_fold_cross_validation(self, X, y, k=5):
        # K折交叉驗證
        fold_size = len(X) // k
        folds_X = [X[i * fold_size:(i + 1) * fold_size] for i in range(k)]
        folds_y = [y[i * fold_size:(i + 1) * fold_size] for i in range(k)]

        accuracies = []
        model_list = []
        for i in range(k):
            # 選擇第i個折疊作為驗證集，其餘作為訓練集
            X_train = np.concatenate([folds_X[j] for j in range(k) if j != i])
            y_train = np.concatenate([folds_y[j] for j in range(k) if j != i])
            X_val = folds_X[i]
            y_val = folds_y[i]

            # 初始化並訓練決策樹模型
            model = PruneDecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features, min_samples_split=self.min_samples_split)
            model.fit(X_train, y_train)

            # 預測驗證集並計算準確度
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            accuracies.append(accuracy)
            model_list.append(model)

            print(f'第 {i + 1} 折的 PruneDecisionTree 準確度：{accuracy}')

        # 計算平均準確度
        avg_accuracy = np.mean(accuracies)
        print(f'平均 PruneDecisionTree 準確度：{avg_accuracy}')
        return model_list

###############################  Prune Decision Tree Classifier  ###############################

###############################  Randon Forest with 2-layer MLP  ###############################
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def train(self, X, y, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backpropagation
            loss = self.cross_entropy_loss(output, y)
            self.backward(X, y, output, learning_rate)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def cross_entropy_loss(self, output, y):
        m = y.shape[0]
        log_likelihood = -np.log(output[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, X, y, output, learning_rate):
        m = y.shape[0]

        # Output layer gradient
        dZ2 = output
        dZ2[range(m), y] -= 1
        dZ2 /= m

        # Backpropagation through the second layer
        dW2 = np.dot(self.a1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Backpropagation through the first layer
        dZ1 = np.dot(dZ2, self.weights2.T)
        dZ1[self.z1 <= 0] = 0  # ReLU derivative
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.weights2 -= learning_rate * dW2
        self.bias2 -= learning_rate * db2
        self.weights1 -= learning_rate * dW1
        self.bias1 -= learning_rate * db1


class DeepRandomForest:
    def __init__(self, num_trees, num_features, num_instances, num_classes):
        self.num_trees = num_trees
        self.num_features = num_features
        self.num_instances = num_instances
        self.num_classes = num_classes
        self.trees = []

    def train(self, X, y, num_epochs=100):
        for _ in range(self.num_trees):
            # Random subset of features and instances
            selected_features = np.random.choice(X.shape[1], self.num_features, replace=False)
            selected_instances = np.random.choice(X.shape[0], self.num_instances, replace=False)
            X_subset = X[selected_instances][:, selected_features]
            y_subset = y[selected_instances]

            # Train an MLP with the correct input size
            mlp = MLP(self.num_features, hidden_size=64, output_size=self.num_classes)
            mlp.train(X_subset, y_subset, epochs=num_epochs)

            # Add the trained MLP to the list of trees
            self.trees.append(mlp)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.num_classes))

        for tree in self.trees:
            # Use each tree to make predictions
            tree_output = tree.forward(X[:, :self.num_features])  # Use only selected features
            predictions += tree_output

        # Combine predictions using simple averaging
        final_predictions = predictions / self.num_trees

        return np.argmax(final_predictions, axis=1)

###############################  Randon Forest with 2-layer MLP  ###############################

###############################  Randon Forest  ###############################
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.feature_subset = None

    def fit(self, X, y, depth=0,  features_subset=None):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            self.label = np.argmax(np.bincount(y))
        else:
            if features_subset is None:
                if self.max_features is not None:
                    self.feature_subset = np.random.choice(X.shape[1], self.max_features, replace=False)
                else:
                    self.feature_subset = np.arange(X.shape[1])
            X_subset = X[:, self.feature_subset]
            self.split_feature, self.split_threshold = self.find_best_split(X_subset, y)
            if self.split_feature is not None:
                left_mask = X[:, self.split_feature] <= self.split_threshold
                right_mask = X[:, self.split_feature] > self.split_threshold
                if np.any(left_mask) and np.any(~left_mask):
                    self.left = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features)
                    self.right = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features)
                    self.left.fit(X[left_mask], y[left_mask], depth + 1)
                    self.right.fit(X[right_mask], y[right_mask], depth + 1)
                else:
                    if np.any(left_mask):
                        self.label = np.argmax(np.bincount(y[left_mask]))
                    else:
                        self.label = np.argmax(np.bincount(y[~left_mask]))
            else:
                self.label = np.argmax(np.bincount(y))

    def find_best_split(self, X, y):
        num_features = X.shape[1]
        best_split_feature = None
        best_split_threshold = None
        best_gini = float('inf')
        
        threshold_value = 0.01
        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])

            if len(unique_values) == 2: 
                thresholds = [unique_values.min(), (unique_values.min() + unique_values.max()) / 2, unique_values.max()]
            elif len(unique_values) <= 10:  # Categorical feature with <= 10 categories
                thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]
            else:
                # For features with more than 10 unique values, use percentiles
                thresholds = np.percentile(unique_values, np.linspace(0, 100, 10))
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                gini_left = self.calculate_gini(y[left_mask])
                gini_right = self.calculate_gini(y[right_mask])

                weighted_gini = (len(y[left_mask]) / len(y)) * gini_left + (len(y[right_mask]) / len(y)) * gini_right
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split_feature = feature
                    best_split_threshold = threshold
                    # Early stopping if the Gini impurity reduction is below a threshold
                    if best_gini < threshold_value:
                        return best_split_feature, best_split_threshold
        
        return best_split_feature, best_split_threshold

    def calculate_gini(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)
    

    def feature_importance(self, X, y):
        if not hasattr(self, 'split_feature'):
            return np.zeros(X.shape[1])  # Return all zeros if tree is not trained yet

        feature_importance = np.zeros(X.shape[1])
        self._accumulate_feature_importance(X, y, feature_importance)
        return feature_importance

    def _accumulate_feature_importance(self, X, y, feature_importance):
        if hasattr(self, 'split_feature'):
            feature_importance[self.split_feature] += self.calculate_gini(y)
            if hasattr(self, 'left'):
                self.left._accumulate_feature_importance(X, y, feature_importance)
                self.right._accumulate_feature_importance(X, y, feature_importance)

    def predict(self, X):
        if hasattr(self, 'split_feature') and self.split_feature is not None:
            is_left = X[:, self.split_feature] <= self.split_threshold
            y_pred = np.zeros(X.shape[0])
            if hasattr(self, 'left'):
                y_pred[is_left] = self.left.predict(X[is_left])
                y_pred[~is_left] = self.right.predict(X[~is_left])
            else:
                y_pred[is_left] = self.label
                y_pred[~is_left] = self.label
            return y_pred
        else:
            return np.full(X.shape[0], self.label)
        
        
    def k_fold_cross_validation(self, X, y, k=5, max_depth=None, max_features=None):
        # 将数据分成K个折叠
        fold_size = len(X) // k
        folds_X = [X[i * fold_size:(i + 1) * fold_size] for i in range(k)]
        folds_y = [y[i * fold_size:(i + 1) * fold_size] for i in range(k)]

        accuracies = []
        model_list = []
        for i in range(k):
            # 选择第i个折叠作为验证集，其余作为训练集
            X_train = np.concatenate([folds_X[j] for j in range(k) if j != i])
            y_train = np.concatenate([folds_y[j] for j in range(k) if j != i])
            X_val = folds_X[i]
            y_val = folds_y[i]

            # 初始化并训练决策树模型
            model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
            model.fit(X_train, y_train)

            # 预测验证集并计算准确度
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            accuracies.append(accuracy)
            model_list.append(model)

            print(f'DecisionTree of Fold {i + 1}: Accuracy = {accuracy}')

        # 计算平均准确度
        avg_accuracy = np.mean(accuracies)
        print(f'DecisionTree of Average Accuracy: {avg_accuracy}')
        return model_list

class RandomForest:
    def __init__(self, num_trees, num_features, num_instances, num_classes):
        self.num_trees = num_trees
        self.num_features = num_features
        self.num_instances = num_instances
        self.num_classes = num_classes
        self.trees = []

    def fit(self, X, y, num_epochs=100):
        for _ in range(self.num_trees):
            # Random subset of features and instances
            selected_features = np.random.choice(X.shape[1], self.num_features, replace=False)
            selected_instances = np.random.choice(X.shape[0], self.num_instances, replace=False)
            X_subset = X[selected_instances][:, selected_features]
            y_subset = y[selected_instances]

            # Train an MLP with the correct input size
            tree = DecisionTreeClassifier()  # 設置樹的最大深度
            tree.fit(X_subset, y_subset)

            # Add the trained MLP to the list of trees
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0]))

        for tree in self.trees:
            # Use each tree to make predictions
            tree_output = tree.predict(X[:, :self.num_features])  # Use only selected features
            predictions += tree_output

        # Combine predictions using simple averaging
        final_predictions = predictions / self.num_trees
        return (final_predictions >= 0.5).astype(int)

###############################  Randon Forest  ###############################
