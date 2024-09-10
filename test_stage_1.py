import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from train import MyDataset
from train import NET


def metrics_stage_1(true_label, predict_label):
    num_samples = predict_label.shape[0]
    ones = np.ones(num_samples)
    # TKR:
    #  A: the number of known samples are accepted / B: the number of known samples
    a = np.sum(np.logical_and(predict_label != (-ones), true_label != (-ones)))
    b = np.sum(true_label != (-ones))
    tkr = a / b

    # TUR:
    #  A: the number of unknown are rejected / B: the number of unknown samples
    a = np.sum(np.logical_and(true_label == (-ones), predict_label == (-ones)))
    b = np.sum(true_label == (-ones))
    tur = a / b

    # KP:
    # A: the number of known samples are accurately classified / the number of all accepted samples
    a = np.sum(true_label[true_label != (-ones)] == predict_label[true_label != (-ones)])
    b = np.sum(predict_label != (-ones))
    if (b == 0):
        kp = 0
    else:
        kp = a / b

    # FKR:
    # A: the number of unknown samples are accurately rejected / the number of all rejected samples
    a = np.sum(true_label[true_label == (-ones)] == predict_label[true_label == (-ones)])
    b = np.sum(predict_label == (-ones))
    fkr = a / b

    # Known Accuracy
    accuracy = []
    for ii in range(num_known):
        a = np.sum(np.logical_and(true_label == (ii * ones), predict_label == (ii * ones)))
        b = np.sum(true_label == (ii * ones))
        if b != 0:
            acc = a / b
            accuracy.append(acc)
        else:
            accuracy.append(-1)
    return tkr, tur, kp, fkr, accuracy


def outlier_check(distance_list):
    distance = np.flip(np.sort(distance_list))
    # print("distance:", distance)
    distance_std = np.std(np.hstack([distance, -distance]))
    threshold = distance[0]
    for index in range(distance.shape[0]):
        threshold = distance[index]
        if distance[index] <= 3 * distance_std:
            break
    return threshold


if __name__ == "__main__":
    tips = '(xyz_for_loss_curve)'
    my_index = 1
    semantic_dim = 128
    expand_semantic_dim = 128
    time_size = 1
    gamma = 0.75
    margin = 8
    num_known = 18
    num_unknown = 6
    num_total = num_known + num_unknown
    model_path = "./model/S3R/group_" + str(my_index) + \
                 "_margin_" + str(margin) + \
                 "_dim_" + str(semantic_dim) + \
                 "_length_" + str(time_size) + \
                 "_gamma_" + str(gamma) + \
                 "_tips_" + tips + \
                 ".pkl"
    model_paras = torch.load(model_path)
    # load data
    known_train_data = MyDataset(path_txt='./experiment_groups/' + str(my_index) + '-known_for_train', len_time=time_size, gamma=gamma, size=512)
    known_test_data = MyDataset(path_txt='./experiment_groups/' + str(my_index) + '-known_for_test', len_time=time_size, gamma=gamma, size=512)
    unknown_test_data = MyDataset(path_txt='./experiment_groups/' + str(my_index) + '-unknown', len_time=time_size, gamma=gamma,size=512)
    known_train_loader = DataLoader(dataset=known_train_data, batch_size=1, shuffle=True, drop_last=False)
    known_test_loader = DataLoader(dataset=known_test_data, batch_size=1, shuffle=True, drop_last=False)
    unknown_test_loader = DataLoader(dataset=unknown_test_data, batch_size=1, shuffle=True, drop_last=False)
    # configure network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Net = NET(in_channels=1, input_size=[int(512*time_size), 512], semantic_dim=semantic_dim, num_class=num_known, device=device).to(device)
    Net.load_state_dict(model_paras)
    # get centers, thresholds from train data
    with torch.no_grad():
        Net.eval()
        # read all semantics of training samples
        train_X = torch.zeros((len(known_train_data), semantic_dim))
        train_Y = torch.zeros(len(known_train_data))
        for i, data in enumerate(tqdm(known_train_loader)):
            x_batch, y_batch, z_batch, label = data
            x_batch, y_batch, z_batch, label = \
                x_batch.to(device), y_batch.to(device), z_batch.to(device), label.to(device)
            _, train_X[i], _, _, _, _ = Net(x_batch, y_batch, z_batch)
            train_Y[i] = label
        theta = torch.zeros(num_known)                                     
        dist_matrix = np.zeros((num_known, semantic_dim, semantic_dim)) 
        class_centers = torch.zeros((num_known, semantic_dim))             
        for clas in range(num_known):
            samples = train_X[train_Y == clas].cpu().numpy()
            covariance_mat = np.cov(samples, rowvar=False, bias=True)
            dist_matrix[clas] = np.linalg.pinv(covariance_mat)
            class_centers[clas] = torch.mean(train_X[train_Y == clas], dim=0)
            x = (train_X[train_Y == clas] - class_centers[clas].expand([samples.shape[0], semantic_dim]))
            x = x.cpu().numpy()
            dist_list = np.sqrt(np.matmul(np.matmul(x, dist_matrix[clas]), np.transpose(x))).diagonal()
            theta[clas] = outlier_check(dist_list)
        # read all testing data
        test_X = torch.zeros((len(known_test_data) + len(unknown_test_data), semantic_dim))
        test_X_expand = torch.zeros((len(known_test_data) + len(unknown_test_data), expand_semantic_dim * 3))
        test_Y = torch.zeros(len(known_test_data) + len(unknown_test_data))
        confusion_matrix = np.zeros((num_known, num_known))
        for i, data in enumerate(tqdm(known_test_loader)):
            x_batch, y_batch, z_batch, label = data
            x_batch, y_batch, z_batch, label = \
                x_batch.to(device), y_batch.to(device), z_batch.to(device), label.to(device)
            predict, test_X[i], _, _, _, test_X_expand[i] = Net(x_batch, y_batch, z_batch)
            test_Y[i] = label
            pre = torch.max(predict.data, 1)[1]
            confusion_matrix[int(label.cpu().numpy())][int(pre.cpu().numpy())] += 1
        for i, data in enumerate(tqdm(unknown_test_loader)):
            x_batch, y_batch, z_batch, label = data
            x_batch, y_batch, z_batch, label = \
                x_batch.to(device), y_batch.to(device), z_batch.to(device), label.to(device)
            _, test_X[len(known_test_data) + i], _, _, _, test_X_expand[len(known_test_data) + i] = Net(x_batch, y_batch, z_batch)
            test_Y[len(known_test_data) + i] = label

        # start evaluation at stage 1
        d_ct = np.zeros((test_X.shape[0], num_known))
        for xi in range(num_known):
            for xj in range(test_X.shape[0]):
                x = (test_X[xj] - class_centers[xi]).cpu().numpy()
                d_ct[xj, xi] = np.sqrt(np.matmul(np.matmul(x, dist_matrix[xi]), np.transpose(x)))
        Theta = theta.expand([test_X.shape[0], num_known]).numpy()
        x_ct = d_ct - Theta
        label_hat = np.zeros(test_X.shape[0])
        tur_mistake = np.zeros((num_total - num_known, num_known))
        for xi in range(test_X.shape[0]):
            if np.min(x_ct[xi]) > 0:
                label_hat[xi] = -1 
            else:
                label_hat[xi] = np.argmin(x_ct[xi])
                if test_Y[xi] >= num_known:
                    tur_mistake[int(test_Y[xi] - num_known), np.argmin(x_ct[xi])] += 1 
        test_Y_normalized = test_Y.cpu().numpy().copy()
        for xi in range(test_Y_normalized.shape[0]):
            if test_Y_normalized[xi] >= num_known:
                test_Y_normalized[xi] = -1 

        tkr, tur, kp, fkr, accuracy = metrics_stage_1(test_Y_normalized, label_hat)
        supervised_acc = []
        for xi in range(num_known):
            supervised_acc.append(confusion_matrix[xi][xi] / np.sum(confusion_matrix[xi]))
        print("supervised accuracy:\n", supervised_acc)
        print("known accuracy:", np.round(accuracy, 4))
        print("mean known accuracy:", np.mean(np.round(accuracy, 4)))
        print("tkr, tur, kp, fkr:", tkr, tur, kp, fkr)
        print("tur mistake:\n", tur_mistake)
        np.save('./test_X.npy', test_X.cpu().numpy())
        np.save('./test_Y.npy', test_Y.cpu().numpy())
        np.save('./label_hat.npy', label_hat)
        np.save('./theta.npy', theta.cpu().numpy())
        np.save('./centers.npy', class_centers.cpu().numpy())
        np.save('./dist_matrix.npy', dist_matrix)
        np.save('./test_X_expand.npy', test_X_expand.cpu().numpy())

        # record semantics
        np.save('./semantic/S3R/test_X.npy', test_X.cpu().numpy())
        np.save('./semantic/S3R/test_Y.npy', test_Y.cpu().numpy())
        np.save('./semantic/S3R/test_X_expand.npy', test_X_expand.cpu().numpy())





