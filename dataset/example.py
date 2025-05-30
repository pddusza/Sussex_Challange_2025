'''
SHLData will provide data as follows:
X: 500x1 size torch tensor in case of singlefile mode, and 500x9 size torch tensor in case of singlefolder mode
   This represents a single row from file/files in selected sensor position
y: 1x1 size torch tensor representing the median value of row in Label.txt file corresponding to time series datapoint
'''
from SHLData import SHLData
from torch.utils.data import DataLoader
shl_data_test = SHLData(root_path='../../data', flag='test', mode='singlefile')
loader = DataLoader(shl_data_test, batch_size=32)
for x in loader:
    print(x)
    break

shl_data_train = SHLData(root_path='../../data', flag='validation', mode='singlefolder')
loader_train= DataLoader(shl_data_train, batch_size=32)
for x, y in loader_train:
    print(x)
    print(y)
    break
