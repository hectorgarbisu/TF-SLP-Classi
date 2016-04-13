__author__ = 'geco'
from os import listdir
from random import shuffle

from matplotlib import pyplot
from copy import deepcopy

class dataset_loader:

    """ path """
    def __init__ (self, path="./dataset/"):
        self.path = path

    "Whenever a class set is under a certain (min_per_label) number of samples" \
    "it will be ignored. Program will also avoid reading over (max_per_label) samples" \
    "of a given class"
    "Each data sample will be adjusted to have exactly (fixed_sig_size) points"
    def load(self, min_per_label=20, max_per_label=40, fixed_sig_size=100):
        file_list = listdir(self.path)
        "get list of available class names"
        labels_dict = self._get_labels(file_list)
        "get cardinality of each class (dictionary)"
        print "labels dict: ",labels_dict
        "Cap every class set to a maximum number of members equal" \
        "to the size of the smaller class (ccty)"
        labels_dict = self._filter_labels(labels_dict,min_per_label,max_per_label)
        print "filtered and adjusted labels: ", labels_dict
        "read (ccty) files of each class from folder at random order" \
        "get files, its labels, and its sizes"
        files,labels,sizes = self._get_files(file_list,labels_dict)
        print "files: ",len(files)," labels: ",labels," sizes: ",sizes
        "Fix every sample to (fixed_sig_size)"
        fixed_data = self._interpolate_points(files,sizes,fixed_sig_size)
        "build tensor of data [label_idx, sample_idx, fixed_sig_size*2]"
        # print len(files[4][0::2]),len(files[4][1::2])
        _, (b,c) = pyplot.subplots(2)
        b.plot(files[4][0::2],files[4][1::2])
        c.plot(fixed_data[4][0::2],fixed_data[4][1::2])
        pyplot.show()

    def _get_labels(self,filelist):
        labels_dict = dict()
        for filename in filelist:
            label = self._label_from_filename(filename)
            try:
                labels_dict[label] = labels_dict[label]+1
            except:
                labels_dict[label] = 1

        return labels_dict

    def _label_from_filename(self,filename):
        return str.split(filename,"-")[0]

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

    def _get_files(self,file_list,labels_dict):
        """ Return a vector of "files", a vector with its labels and a vector of sizes """
        shuffle(file_list)
        i = 0
        files,labels,sizes = [[],[],[]]
        for filename in file_list:
            label = self._label_from_filename(filename)
            if (label in labels_dict):
                " open and read the file " \
                " must check that label belongs in labels_dict"
                sample_points, period, size = self._read_points(filename)
                files.append(sample_points)
                " label is inserted as is "
                labels.append(label)
                " after opening the file, see how many points it contains"
                sizes.append(size)
                i += 1
        return files,labels,sizes

    def _read_points(self,filename):
        f = open(self.path+"/"+filename,'r')
        period = f.readline()
        xdim, ydim = f.readline().split()
        _ = f.readline()
        data = list()
        for line in f:
            xi,yi = line.split()
            "coordinates between -1 and 1"
            data.append(2*float(xi)/float(xdim)-1)
            data.append(1-2*float(yi)/float(ydim))
        dataset_length = len(data)/2
        f.close()
        return data,period,dataset_length

    def _interpolate_points(self,data_samples,sizes,number_of_points):
        # TODO: deepcopy isn't enough
        new_points = self._l_of_l_copy(data_samples)
        print len(new_points)
        for ii in range(len(sizes)):
            size = sizes[ii]
            # print size, len(data_samples[ii])
            if(size>number_of_points):
                # TODO : this
                "We need to remove some points"
                residue = number_of_points-size
                remove_distance = size//residue
                new_points[ii].remove(remove_distance-1)
            elif(size<1):
                "This signature sucks"
                Exception("Signature without data")
            else:
                "We need to add some points"
                "Add the middle point of the next pair of points"
                "until size=number_of_points"
                current_size = size
                jj = 0
                while(current_size<number_of_points):
                    new_x,new_y = self._middle_point(new_points[ii][jj],new_points[ii][jj+1],
                                                     new_points[ii][jj+2],new_points[ii][jj+3])
                    new_points[ii].insert(jj+1,new_x)
                    new_points[ii].insert(jj+3,new_y)
                    current_size += 1
                    " loop through the whole sample as many times as needed "
                    jj = (jj+4)%(current_size)
        return new_points

    def _middle_point(self,x1,y1,x2,y2):
        return (x1+x2)/2,(y1+y2)/2

    def _l_of_l_copy(self, list_of_lists):
        " Yo dawg "
        copy = list()
        for sublist in list_of_lists:
            sublist_copy = list()
            for element in sublist:
                sublist_copy.append(deepcopy(element))
            copy.append(sublist_copy)
        return copy

    def next_batch(self, batch_size):
        print "miau"