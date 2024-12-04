import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from contLoss import ContLoss
import torch.optim as optim
import numpy as np
import os


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

    return tkr, tur, kp, fkr


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


def position_coding(x):
    num_token, num_dims = x.size(-2), x.size(-1)
    p = torch.zeros((1, num_token, num_dims))
    t = torch.arange(num_token, dtype=torch.float32).reshape(-1, 1) / \
        torch.pow(1e4, torch.arange(0, num_dims, 2, dtype=torch.float32) / num_dims)
    p[:, :, 0::2] = torch.sin(t)
    p[:, :, 1::2] = torch.cos(t)
    return p


class MyDataset(Dataset):
    def __init__(self, path_txt, len_time, gamma, size=512):
        super(MyDataset, self).__init__()
        fh = open(path_txt, 'r')
        data = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            data.append((words[0], int(words[1])))
        self.data = data
        self.size = size
        self.gamma = gamma
        self.time = int(len_time * self.size)

    def __getitem__(self, index):
        path, label = self.data[index]
        x = np.load(path + '-a.npy')
        # resize feature into size 512
        x = x[0:self.time, :]
        x = x[:, 0:self.size]
        x = torch.FloatTensor(x)
        z = x.permute(1, 0)
        return x.unsqueeze(0), x, z, label

    def __len__(self):
        return len(self.data)


class NET(nn.Module):
    def __init__(self, in_channels, input_size, semantic_dim, num_class, device):
        super(NET, self).__init__()
        self.input_size = input_size                     # [T, W]
        self.semantic_dim = semantic_dim
        self.num_class = num_class
        self.device = device
        self.in_channels = in_channels
        self.SA_1 = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dim_feedforward=256)
        self.SA_1 = nn.TransformerEncoder(self.SA_1, num_layers=3)
        self.SA_2 = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dim_feedforward=256)
        self.SA_2 = nn.TransformerEncoder(self.SA_2, num_layers=3)

        self.encoding_to_sa1 = nn.Sequential(
            nn.Linear(self.input_size[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.encoding_to_sa2 = nn.Sequential(
            nn.Linear(self.input_size[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.sa1_to_semantic = nn.Sequential(
            nn.Linear(int(self.input_size[0] * 64), 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )

        self.sa2_to_semantic = nn.Sequential(
            nn.Linear(int(self.input_size[1] * 64), 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )

        self.total_semantic = nn.Sequential(
            nn.Linear(self.semantic_dim * 3, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU(),
            nn.Linear(self.semantic_dim, self.semantic_dim, bias=False),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )
        self.encoder_d1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.encoder_d3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.encoder_d5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.encoder_to_semantic = nn.Sequential(
            nn.Linear(128*3, self.semantic_dim*2),
            nn.BatchNorm1d(self.semantic_dim*2),
            nn.ReLU(),
            nn.Linear(self.semantic_dim*2, self.semantic_dim),
            nn.BatchNorm1d(self.semantic_dim),
            nn.ReLU()
        )
        self.semantic_to_classifier = nn.Sequential(
            nn.Linear(self.semantic_dim, self.num_class)
        )

    def forward(self, x, y, z):
        encoder1 = self.encoder_d1(x)
        encoder2 = self.encoder_d3(x)
        encoder3 = self.encoder_d5(x)
        encoder_output = torch.cat([encoder1, encoder2, encoder3], dim=1)
        x = F.adaptive_avg_pool2d(encoder_output, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.encoder_to_semantic(x)
        expand_x = F.adaptive_avg_pool2d(encoder_output, (1, 1))
        expand_x = expand_x.view(expand_x.shape[0], -1)
        y = self.encoding_to_sa1(y)                 # [B, T, 64]
        z = self.encoding_to_sa2(z)                 # [B, W, 64]
        y = y + position_coding(y).to(self.device)
        z = z + position_coding(z).to(self.device)
        y = self.SA_1(y)                            # [B, T, 64]
        z = self.SA_2(z)                            # [B, W, 64]
        y = y.view(y.shape[0], -1)                  # [B, T*64]
        z = z.view(z.shape[0], -1)                  # [B, W*64]
        y = self.sa1_to_semantic(y)                 # [B, T*64] -> [B, semantic dim]
        z = self.sa2_to_semantic(z)                 # [B, W*64] -> [B, semantic dim]
        semantic = torch.cat([x, y], dim=1)
        semantic = torch.cat([semantic, z], dim=1)
        semantic = self.total_semantic(semantic)  # [B, 128*3] -> [B, 128]
        predict = self.semantic_to_classifier(semantic)

        return predict, semantic, x, y, z, expand_x


def train(net, device, semantic_dims, lr, batch_size, margin, num_known_class, my_index, gamma, len_time, tips):
    train_data = MyDataset(path_txt='./experiment_groups/' + str(my_index) + '-known_for_train', len_time=len_time, gamma=gamma, size=512)
    test_data_known = MyDataset(path_txt='./experiment_groups/' + str(my_index) + '-known_for_test', len_time=len_time, gamma=gamma, size=512)
    test_data_unknown = MyDataset(path_txt='./experiment_groups/' + str(my_index) + '-unknown', len_time=len_time, gamma=gamma, size=512)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader_for_evaluation = DataLoader(dataset=train_data, batch_size=1, shuffle=True, drop_last=False)
    test_loader_known = DataLoader(dataset=test_data_known, batch_size=1, shuffle=True, drop_last=False)
    test_loader_unknown = DataLoader(dataset=test_data_unknown, batch_size=1, shuffle=True, drop_last=False)
    print("Load train samples : {} | test known samples: {} | test unknown samples: {}"
          .format(len(train_data), len(test_data_known), len(test_data_unknown)))
    # configure loss_function
    ce_loss = torch.nn.CrossEntropyLoss()
    cont_loss = ContLoss(num_classes=num_known_class, device=device, feat_dims=semantic_dims, margin=margin)
    # configure optimizer
    optimizer_net = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4, eps=1e-6)
    optimizer_center = torch.optim.Adam(cont_loss.parameters(), lr=lr, weight_decay=1e-4, eps=1e-6)
    # configure loss weights
    [eta_1, eta_2, eta_3] = [0.05, 1, 0.1]
    # configure lr_scheduler
    lr_scheduler_net = optim.lr_scheduler.StepLR(optimizer_net, step_size=10, gamma=0.98)
    lr_scheduler_center = optim.lr_scheduler.StepLR(optimizer_center, step_size=10, gamma=0.98)

    # ------------------------------START LEARNING---------------------------------#
    max_epoch = 251
    interval = 50
    loss_log = []
    indicator_log = []
    print("START LEARNING !!")
    model_name = 'group_' + str(my_index) + \
                 '_margin_' + str(margin) + \
                 '_dim_' + str(semantic_dims) + \
                 '_length_' + str(len_time) + \
                 '_gamma_' + str(gamma) + \
                 '_tips_' + str(tips)
    for epoch in range(max_epoch+1):
        Loss_ce, Loss_center, Loss_cluster, Loss_total = 0, 0, 0, 0
        net.train()
        # model learning start--------------------
        for x_batch, y_batch, z_batch, label in tqdm(train_loader):
            x_batch, y_batch, z_batch, label = \
                x_batch.to(device), y_batch.to(device), z_batch.to(device), label.to(device)
            # predict, semantic, x, y, z, expand_x = net(x_batch)
            predict, semantic, _, _, _, _ = net(x_batch, y_batch, z_batch)
            loss_center, loss_cluster = cont_loss(semantic, label)
            loss_ce = ce_loss(predict, label)
            loss_total = eta_1 * loss_center + eta_2 * loss_cluster + eta_3 * loss_ce
            # grads backward
            optimizer_net.zero_grad()
            optimizer_center.zero_grad()
            loss_total.backward()
            optimizer_net.step()
            optimizer_center.step()
            # record loss
            Loss_center += loss_center * eta_1
            Loss_cluster += loss_cluster * eta_2
            Loss_ce += loss_ce * eta_3
            Loss_total += loss_total
        print("Epoch {}: "
              "\n Total: {:.4f} | CE: {:.4f} | Center:{:.4f} | Cluster:{:.4f}"
              .format(epoch, Loss_total, Loss_ce, Loss_center, Loss_cluster))
        # lr decrease
        lr_scheduler_net.step()
        lr_scheduler_center.step()
        # model learning end-----------------------
        loss_log.append([epoch, Loss_total.item(), Loss_center.item(), Loss_ce.item(), Loss_cluster.item()])
        np.savetxt('./model/S3R/' + model_name + '_loss' + '.txt', loss_log)

        if epoch % interval == 0 and epoch != 0:
            print("----------------------start evaluation----------------------")
            # get centers, thresholds from train data
            with torch.no_grad():
                net.eval()
                # read all semantics of training samples
                train_X = torch.zeros((len(train_data), semantic_dim))
                train_Y = torch.zeros(len(train_data))
                for i, data in enumerate(tqdm(train_loader_for_evaluation)):
                    x_batch, y_batch, z_batch, label = data
                    x_batch, y_batch, z_batch, label = \
                        x_batch.to(device), y_batch.to(device), z_batch.to(device), label.to(device)
                    _, train_X[i], _, _, _, _ = net(x_batch, y_batch, z_batch)
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
                test_X = torch.zeros((len(test_data_known) + len(test_data_unknown), semantic_dim))
                test_Y = torch.zeros(len(test_data_known) + len(test_data_unknown))
                confusion_matrix = np.zeros((num_known, num_known))
                for i, data in enumerate(tqdm(test_loader_known)):
                    x_batch, y_batch, z_batch, label = data
                    x_batch, y_batch, z_batch, label = \
                        x_batch.to(device), y_batch.to(device), z_batch.to(device), label.to(device)
                    predict, test_X[i], _, _, _, _ = net(x_batch, y_batch, z_batch)
                    test_Y[i] = label
                    pre = torch.max(predict.data, 1)[1]
                    confusion_matrix[int(label.cpu().numpy())][int(pre.cpu().numpy())] += 1
                for i, data in enumerate(tqdm(test_loader_unknown)):
                    x_batch, y_batch, z_batch, label = data
                    x_batch, y_batch, z_batch, label = \
                        x_batch.to(device), y_batch.to(device), z_batch.to(device), label.to(device)
                    _, test_X[len(test_data_known) + i], _, _, _, _ = net(x_batch, y_batch, z_batch)
                    test_Y[len(test_data_known) + i] = label

                # start evaluation at stage 1
                d_ct = np.zeros((test_X.shape[0], num_known))
                for xi in range(num_known):
                    for xj in range(test_X.shape[0]):
                        x = (test_X[xj] - class_centers[xi]).cpu().numpy()
                        d_ct[xj, xi] = np.sqrt(np.matmul(np.matmul(x, dist_matrix[xi]), np.transpose(x)))
                theta = theta.expand([test_X.shape[0], num_known]).numpy()
                x_ct = d_ct - theta
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

                tkr, tur, kp, fkr = metrics_stage_1(test_Y_normalized, label_hat)
                print("----------------------evaluation end----------------------")
                torch.save(net.state_dict(), './model/S3R/' + model_name + '.pkl')
                print("current state epoch: {} | tkr: {} | tur: {} | kp: {}".format(epoch, tkr, tur, kp))
                print("tur mistake:", tur_mistake)
                indicator_log.append([epoch, np.round(tkr, 4), np.round(tur, 4), np.round(kp, 4), np.round(fkr, 4)])
        np.savetxt('./model/S3R/' + model_name + '_indicator' + '.txt', indicator_log)


if __name__ == "__main__":
    tips = '(xyz_for_loss_curve)'
    my_index = 1
    len_time = 1
    gamma = 0.75
    margin = 8
    num_known = 18
    num_unknown = 6
    semantic_dim = 128
    num_total = num_known + num_unknown
    if not os.path.exists('./model/S3R/'):
        os.makedirs('./model/S3R/')
    if not os.path.exists('./semantic/S3R/'):
        os.makedirs('./semantic/S3R/')    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Net = NET(in_channels=1, input_size=[int(512 * len_time), 512], semantic_dim=semantic_dim,
              num_class=num_known, device=device).to(device)
    train(net=Net, device=device, semantic_dims=semantic_dim, lr=1e-4, batch_size=32
          , margin=margin, num_known_class=num_known, my_index=my_index, gamma=gamma, len_time=len_time, tips=tips)


