from ts2vec import TS2Vec
import datautils
import pandas as pd
import numpy as np
import os

dir = './UCR_Anomaly_FullData/'
list = os.listdir(dir)
train_data = []
test_data = []
anomaly_record = []
for i in range(0, len(list)):
    path = os.path.join(dir, list[i])
    (filename, _) = os.path.splitext(list[i])
    args = filename.split('_')
    if args[-4][-6:] == 'sddb40' and args[-4][0:9] == 'DISTORTED':
        train_len, anomaly_begin, anomaly_end = int(args[-3]), int(args[-2]), int(args[-1])
        data = np.loadtxt(path).tolist()
        train_data.append(data[0:train_len])
        test_data.append(data[train_len:79000])
        anomaly_record.append([anomaly_begin, anomaly_end])
train_data = np.array(train_data)[:,:,np.newaxis]
test_data = np.array(test_data)[:,:,np.newaxis]

# (Both train_data and test_data have a shape of n_instances x n_timestamps x n_features)

# Train a TS2Vec model
model = TS2Vec(
    input_dims=1,
    device=0,
    output_dims=320
)
loss_log = model.fit(
    train_data,
    verbose=True
)
# # Compute timestamp-level representations for test set
# test_repr = model.encode(test_data)  # n_instances x n_timestamps x output_dims
#
# # Compute instance-level representations for test set
# test_repr = model.encode(test_data, encoding_window='full_series')  # n_instances x output_dims

# Sliding inference for test set
test_repr = model.encode(
    test_data,
    casual=True,
    sliding_length=1,
    sliding_padding=1000
)  # n_instances x n_timestamps x output_dims
# (The timestamp t's representation vector is computed using the observations located in [t-50, t])
np.save("DISTORTEDsddb40_repr.npy",test_repr)