__author__ = 'geco'
import SLP
from dataset_loader import dataset_loader as dl

dataset_path = "./dataset"
num_epochs = 20000
# dataset_lenght = 100
sample_size = 400
num_batches = 1
alpha = 0.01
nW_hidden = 5

slp = SLP.SLP(sample_size*2,nW_hidden,2)
dl(dataset_path).load()