from ts2vec import TS2Vec
import datautils
import numpy as np
import os

# dir = './UCR_Anomaly_FullData/'
# list = os.listdir(dir)
# train_data = []
# test_data = []
# anomaly_record = []
# for i in range(0, len(list)):
#     path = os.path.join(dir, list[i])
#     (filename, _) = os.path.splitext(list[i])
#     args = filename.split('_')
#     if args[-4][-6:] == 'sddb40' and args[-4][0:9] == 'DISTORTED':
#         train_len, anomaly_begin, anomaly_end = int(args[-3]), int(args[-2]), int(args[-1])
#         data = np.loadtxt(path).tolist()
#         train_data.append(data[0:train_len])
#         test_data.append(data[train_len:79000])
#         anomaly_record.append([anomaly_begin, anomaly_end])
# train_data = np.array(train_data)[:,:,np.newaxis]
# test_data = np.array(test_data)[:,:,np.newaxis]

def load_train_data(len_l, len_r, window_size=1000, dir='./UCR_Anomaly_FullData/'):
    list = os.listdir(dir)
    train_x = []
    train_y = []
    label_dict = {}
    dataset_used = []
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        (filename, _) = os.path.splitext(list[i])
        label_dict[i] = filename
        args = filename.split('_')
        train_len = int(args[-3])
        if len_l < train_len <= len_r:
            dataset_used.append(i)
            data = np.loadtxt(path).tolist()
            data = data[0:train_len]
            for j in range(0, len(data)-window_size, window_size):
                train_x.append(data[i:i+window_size])
                train_y.append(i)
    train_x = np.array(train_x)[:, :, np.newaxis]
    train_y = np.array(train_y)
    return train_x, train_y, label_dict, dataset_used

def load_test_data(label, label_dict, dir='./UCR_Anomaly_FullData/'):
    list = os.listdir(dir)
    test_data = []
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        (filename, _) = os.path.splitext(list[i])
        args = filename.split('_')
        if filename==label_dict[label]:
            train_len = int(args[-3])
            data = np.loadtxt(path).tolist()
            test_data = data[train_len:]
    test_data = np.array(test_data)[np.newaxis, :, np.newaxis]
    return test_data


train_x , train_y, label_dict, dataset_used= load_train_data(0,5000,window_size=1000)
train_data = train_x

# (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)

# Train a TS2Vec model
model = TS2Vec(
    input_dims=1,
    device=1,
    batch_size=16,
    output_dims=320
)
loss_log = model.fit(
    train_data,
    n_epochs=200,
    verbose=True
)

save_path = 'repr/epoch200_batchsize16_outputdim320_padding200_len0_5000_withoutlabel'
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i in dataset_used:
    test_data = load_test_data(i , label_dict, dir='./UCR_Anomaly_FullData/')
    test_repr = model.encode(
        test_data,
        casual=True,
        sliding_length=1,
        sliding_padding=200
    )  # n_instances x n_timestamps x output_dims
    # (The timestamp t's representation vector is computed using the observations located in [t-sliding_padding, t])
    np.save(os.path.join(save_path,label_dict[i]+".npy"),test_repr)

