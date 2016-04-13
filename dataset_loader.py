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
    def load(self, min_per_label=20, max_per_label=40, fixed_sig_size=400):
        filelist = listdir(self.path)
        "get list of available class names"
        labels_dict = self._get_labels(filelist)
        "get cardinality of each class (dictionary)"
        print "labels dict: ",labels_dict
        "Cap every class set to a maximum number of members equal" \
        "to the size of the smaller class (ccty)"
        labels_dict = self._filter_labels(labels_dict,min_per_label,max_per_label)
        print "filtered and adjusted labels: ", labels_dict
        "read (ccty) files of each class from folder at random order"
        files,sizes = _get_files(filelist,labels_dict)
        "Fix every sample to (fixed_sig_size)"

        "build tensor of data [label_idx, sample_idx, fixed_sig_size*2]"

    def _get_labels(self,filelist):
        labels_dict = dict()
        for filename in filelist:
            label = str.split(filename,"-")[0]
            try:
                labels_dict[label] = labels_dict[label]+1
            except:
                labels_dict[label] = 1

        return labels_dict

    def _filter_labels(self,labels_dict,min_per_label,max_per_label):
        labels_dict_copy = labels_dict.copy()
        min_val = max_per_label
        for label,ctty in labels_dict_copy.iteritems():
            if (ctty<min_per_label):
                "If a class doesn't have enough members it will be ignored"
                print "class ",label,"is deleted because: ",ctty,"<",min_per_label
                labels_dict.pop(label)
            else:
                min_val = min(min_val,ctty)
        for label,ctty in labels_dict.iteritems():
            "If a class does have too many members it will be cut"
            print "class ",label,"is capped from ",ctty,"to ",min_val
            labels_dict[label]=min_val
        return labels_dict
    def _get_files(self,filelist,labels_dict):

        return 1,1

    def next_batch(self, batch_size):
        print "miau"