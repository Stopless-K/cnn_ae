import pandas as pd
import numpy as np
frame = pd.read_csv('cnn_train.csv', sep=',', header=0, index_col=0)
data = frame.values
print(data.shape)
np.savetxt('cnn_train_no_header.csv', data, delimiter=',')



