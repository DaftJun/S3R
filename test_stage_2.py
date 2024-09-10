import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics


def outlier_check(distance_list):
    distance = np.flip(np.sort(distance_list))
    distance_std = np.std(np.hstack([distance, -distance]))
    threshold = distance[0]
    for index in range(distance.shape[0]):
        threshold = distance[index]
        if distance[index] <= 3 * distance_std:
            break
    return threshold


if __name__ == "__main__":
    num_known = 18
    num_unknown = 6
    test_X = np.load('./test_X.npy')
    test_X_expand = np.load('./test_X_expand.npy')
    test_X = np.hstack((test_X, test_X_expand))
    test_Y = np.load('./test_Y.npy')
    label_hat = np.load('./label_hat.npy')
    theta = np.load('./theta.npy')
    class_centers = np.load('./centers.npy')
    dist_matrix = np.load('./dist_matrix.npy')

    test_Y_normalized = test_Y.copy()
    for xi in range(test_Y_normalized.shape[0]):
        if test_Y_normalized[xi] >= num_known:
            test_Y_normalized[xi] = -1

    predict_unknown_X = torch.FloatTensor(test_X[label_hat == -1])
    predict_unknown_Y = torch.LongTensor(test_Y[label_hat == -1])
    ones = np.ones(test_Y.shape[0])
    # if u = 1 ?
    # first calculate the theta for all predict_unknown_X
    samples = predict_unknown_X.cpu().numpy()
    covariance_mat = np.cov(samples, rowvar=False, bias=True)
    matrix = np.linalg.pinv(covariance_mat)
    centers = torch.mean(predict_unknown_X, dim=0)
    x = (predict_unknown_X - centers.expand([samples.shape[0], samples.shape[1]]))
    x = x.cpu().numpy()
    dist_list = np.sqrt(np.matmul(np.matmul(x, matrix), np.transpose(x))).diagonal()
    theta_u1 = outlier_check(dist_list)
    if theta_u1 <= np.max(theta):
        print("u = 1")
        u = 1
        # calculate unknown accuracy
        a = np.sum(np.logical_and(test_Y_normalized == (-ones), label_hat == (-ones)))
        b = np.sum(test_Y_normalized == (-ones))
        unknown_acc = a / b
        print("unknown_acc:", unknown_acc)
        # calculate UP
        c = predict_unknown_X.shape[0]
        UP = a / c
        print("UP:", UP)
    else:
        scalar = MinMaxScaler()
        predict_unknown_X = scalar.fit_transform(predict_unknown_X)
        DB = []
        SC = []
        for ui in range(2, 15):
            Cluster = KMeans(n_clusters=ui, init='k-means++', random_state=51).fit(predict_unknown_X)
            pre_label = Cluster.labels_
            sc = metrics.silhouette_score(predict_unknown_X, pre_label)
            db = metrics.davies_bouldin_score(predict_unknown_X, pre_label)
            print(ui, sc, db)
            DB.append(db)
            SC.append(sc)
        DB = np.array(DB)
        SC = np.array(SC)
        u = np.argmin(DB) + 2
        print("u=", u)
        Cluster = KMeans(n_clusters=u, init='k-means++', random_state=51).fit(predict_unknown_X)
        pred_label = Cluster.labels_
        confusion_mat = np.zeros((u, num_known+num_unknown))
        for xi in range(predict_unknown_X.shape[0]):
            confusion_mat[int(pred_label[xi])][int(predict_unknown_Y[xi])] += 1
        confusion_mat = confusion_mat[:, num_known:]
        print("confusion_mat:", confusion_mat)
        dominate_sample = np.zeros(num_unknown)
        for row in range(u):
            for col in range(num_unknown):
                if confusion_mat[row][col] >= np.sum(confusion_mat[row]) * 0.5 and np.argmax(confusion_mat[:, col]) == row:
                    dominate_sample[col] = confusion_mat[row][col]
        print("dominate_sample:", dominate_sample)
        # calculate the unknown accuracy
        unknown_acc = []
        for clas in range(num_unknown):
            a = np.sum(test_Y == int(clas+num_known) * ones)
            unknown_acc.append(dominate_sample[clas] / a)
        print("unknown_acc:", unknown_acc)
        print("mean unknown_acc:", np.mean(unknown_acc))
        up = np.sum(dominate_sample) / predict_unknown_X.shape[0]
        print("up:", up)








