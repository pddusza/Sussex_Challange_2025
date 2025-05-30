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