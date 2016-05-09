__author__ = 'geco'
import SLP
import numpy as np
from dataset_loader import dataset_loader

dataset_path = "./dataset/"
num_epochs = 20000
# dataset_lenght = 100
sample_size = 100
num_batches = 1
alpha = 0.01
nW_hidden = 5
batch_size = 5

dl = dataset_loader(dataset_path)
dl.load(fixed_sig_size=sample_size)
classes = dl.get_classes()
num_classes = len(classes)
batch,labels = dl.next_2d_batch(batch_size)
# batch2,labels2 = dl.next_2d_batch(batch_size)
# Number of inputs will depend on the adjusted size of each sample
# Number of outputs will depend on the number of different classes to classify
slp = SLP.SLP(sample_size*2,nW_hidden,num_classes)
slp.feed_batch(batch,labels)
print np.array(batch).shape
print labels
