import numpy as np
from collections import OrderedDict
import csv

'''
External data file processing class.
Includes class methods and various forms of processed data
Author: Woohyuk Jang, wjang@cs.yorku.ca
        Professor Brubaker's csv_read

See nbc_binclass_dirmult.py for applications
'''


class Data:
    filename = ''
    seed = True             # if seed true, then training-data consistent for program run
    data_processed = False

    ndims = 0               # class+feature dimensions
    npts = 0                # number of data points
    d_mat = None            # initial processed data matrix
    d_train = None          # training data set
    d_test = None           # test data set
    d_ranges = None         # stores range of feature value f_i's
    char_maps = None        # ordered_dict of f_i's chars and numeric map
    reverse_maps = None     # reverse maps numeric number to f_i's char

    def __init__(self, _filename, _seed=True):
        """
        constructor
        :param _filename: name of file
        :param _seed: seed true or false; DEFAULT TRUE
        """

        self.filename = _filename
        self.seed = _seed

    def setfilename(self, _filename):
        """
        changes file name
        :param _filename: name of file
        """
        self.filename = _filename

    def setseed(self, _seed):
        """
        sets seed to true or false
        :param _seed: seed or not for random training data sets for each run
        """
        self.seed = _seed

    def printstate(self):
        """
        prints state of data object
        """
        print "filename: " + self.filename
        print "seed: " + str(self.seed)
        print "data_processed: " + str(self.data_processed)

    def textcat_csv_read(self, _delimiter=',', perm_train_perc=8):
        """
        reads in data from a csv and processes it into a numerical data object
        ***Can only be used on binomial class, multinomial features, class and feature values
        ***are expressed chars or strings in the input data file
        :param _delimiter: delimiter used in the input file
        :param perm_train_perc: percent of data read that should go into training data set
        """

        # open and read raw data
        _filename = self.filename
        with open(_filename, 'rb') as raw_file:
            raw_data = csv.reader(raw_file, delimiter=_delimiter, quoting=csv.QUOTE_NONE)
            data_list = list(raw_data)

        # number of column dimensions including class label
        self.ndims = len(data_list[0])
        # number of row X_i data points
        self.npts = len(data_list)

        # mappings to make sense of numeric f_i's
        char_maps = [OrderedDict() for i in range(self.ndims)]
        reverse_maps = [[] for i in range(self.ndims)]
        d_mat = np.empty((self.npts, self.ndims), dtype=np.int32)
        for i, cdata in enumerate(data_list):
            for j, cstr in enumerate(cdata):
                # new cstr found for feature_i's OrderedDict#j
                if cstr not in char_maps[j]:
                    # OrderedDict#j (cstr : int (current fullness of OrderedDict#j) )
                    char_maps[j][cstr] = len(char_maps[j])
                    # non-int cstr record for each feature_i
                    reverse_maps[j].append(cstr)
                # fill d_mat[] with correspondent number of feature_i's {N}
                d_mat[i, j] = char_maps[j][cstr]
        del data_list

        if self.seed:
            np.random.seed(0)
        # store results in data object's attributes
        self.char_maps = char_maps
        self.reverse_maps = reverse_maps
        self.d_mat = d_mat
        data_perm = np.random.permutation(self.npts)
        self.d_train = d_mat[data_perm[0:(perm_train_perc * self.npts / 10)], :]
        self.d_test = d_mat[data_perm[(perm_train_perc * self.npts / 10):], :]
        # ***REMINDER: d_ranges columns include class labels for dimension consistency
        self.d_ranges = d_mat[:, :].max(axis=0)
        self.data_processed = True
