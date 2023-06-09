import torch
from torch.utils.data import IterableDataset
from glob import glob

class BallerDataset(IterableDataset):
    def __init__(self, datasetDir):
        self.datasetDir = datasetDir
        
    def __iter__(self):
        return self.it()
    
    def it(self):
        for datasetIn, datasetOut in zip(sorted(glob(self.datasetDir+'/*in.pt')), sorted(glob(self.datasetDir+'/*out.pt'))):
            # print(f'Opening {datasetIn}')
            if datasetIn.rstrip('_in.pt') != datasetOut.rstrip('_out.pt'):
                raise Exception(f"File name mismatch In:{datasetIn}, Out:{datasetOut}")
                
            with open(datasetIn, 'rb') as fin, open(datasetOut, 'rb') as fout:
                x = torch.load(fin)
                y = torch.load(fout)
                for i in range(x.shape[0]):
                    yield x[i], y[i]