__author__ = 'geco'
from os import listdir

class dataset_loader:

    """ path """
    def __init__ (self, path="./dataset"):
        self.path = path

    "Whenever a class set is under a certain (min_per_label) number of samples" \
    "it will be ignored. Program will also avoid reading over (max_per_label) samples" \
    "of a given class"
    "Each data sample will be adjusted to have exactly (fixed_sig_size) points"
    def load(self, min_per_label=20, max_per_label=30, fixed_sig_size=400):
        filelist = listdir(self.path)
        "get list of available class names"
        labels = self._get_labels(filelist)
        "get cardinality of each class"

        "Crop every class set to a maximum number of members equal" \
        "to the size of the smaller class"

        "Fix every sample to (fixed_sig_size)"

        "build tensor of data [label_idx, sample_idx, fixed_sig_size*2]"

    def _get_labels(self,filelist):
        "Each label happens just once"
        labels_set = set([str.split(filename,"-")[0] for filename in filelist])
        return list(labels_set)

    def next_batch(self, batch_size):
        print "miau"