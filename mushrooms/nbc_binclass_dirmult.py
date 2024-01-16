import numpy as np
import matplotlib.pyplot as plt
import operator

'''
Naive Bayes Classifier (Beta Bernoulli; Multinomial Dirilecht).
Includes class methods and various processed data
Author: Woohyuk Jang, wjang@cs.yorku.ca

See data.py for data read and format procedures
'''


class NBCBinClassDirMult:
    data = None                 # Data object
    train_freq_list = None      # training data frequency count
    train_raw_list = None       # training data raw list (for histograms)
    train_status = False        # trained status
    num_class_list = None       # P or E indexer
    num_train_class1 = 0        # number of P
    num_train_class0 = 0        # number of E
    num_train_data = 0          # total number of training data

    def __init__(self, data_obj):
        """
        constructor
        :param data_obj: from data.py (read form of data)
        """
        self.data = data_obj
        np.seterr(divide='ignore')

    def printstate(self):
        """
        prints state of NBC object
        """

        print "nbc_trained: " + str(self.train_status)

    def train_data(self):
        """
        trains from input Data object
        fills in number of training data, number of P, number of E,
        and also initializes and fills in training frequency list
        and training raw list.
        """

        self.num_train_data = self.data.d_train.shape[0]
        self.num_train_class1 = self.data.d_train[:, 0].sum(axis=0)
        self.num_train_class0 = self.num_train_data - self.num_train_class1
        self.num_class_list = (self.num_train_class0, self.num_train_class1)
        print "Number of training data: " + str(int(self.num_train_data))
        print "Fraction of training data that is CLASS ONE: " + str(float(self.num_train_class1) / self.num_train_data)
        print "Fraction of training data that is CLASS ZERO: " + str(float(self.num_train_class0) / self.num_train_data)

        # training data frequency count array [ [P], [E] ] initialized with 0 counts
        self.train_freq_list = [[[0] * (self.data.d_ranges[p] + 1) for p in range(self.data.ndims)],
                                [[0] * (self.data.d_ranges[e] + 1) for e in range(self.data.ndims)]]
        # training data raw list for histograms; same format as above
        self.train_raw_list = [[[] for p in range(self.data.ndims)],
                               [[] for e in range(self.data.ndims)]]
        # for each row of training data, for each feature in the row
        # increment respective frequency count, and append to raw list for histograms
        for x, row_array in enumerate(self.data.d_train):
            for feat, feature_i in enumerate(row_array):
                self.train_freq_list[row_array[0]][feat][feature_i] += 1
                self.train_raw_list[row_array[0]][feat].append(feature_i)
        # trained status now true
        self.train_status = True

    def generate_hists(self, str_zero, str_one, direc='imgs'):
        """
        Generates histograms from frequency count list
        :param str_zero: Name of first class (Poisonous)
        :param str_one: Name of second class (Edible)
        :param direc: desired directory ***DO NOT INCLUDE '/'
        """

        print 'Generating histograms...'
        for i in range(1, self.data.ndims):
            xlabels = []
            for r in range(self.data.d_ranges[i] + 1):
                xlabels.append(self.data.reverse_maps[i][r] + ' ' + '(' + str(r) + ')')

            fig, axarr = plt.subplots(2, sharex=True)
            axarr[0].set_title('Feature ' + str(i) + ', ' + str_zero)
            axarr[0].hist(self.train_raw_list[0][i],
                          bins=np.arange(self.data.d_ranges[i] + 1 + 1) - 0.5,
                          normed=1, facecolor='red', alpha=0.8)
            axarr[0].set_xticks(range(self.data.d_ranges[i] + 1))
            axarr[0].set_xlim([-0.5, self.data.d_ranges[i] + 1 - 0.5])
            axarr[0].set_ylim([0, 1.0])
            axarr[1].set_title(str_one)
            axarr[1].hist(self.train_raw_list[1][i],
                          bins=np.arange(self.data.d_ranges[i] + 1 + 1) - 0.5,
                          normed=1, facecolor='blue', alpha=0.8)
            axarr[1].set_xticks(range(self.data.d_ranges[i] + 1))
            axarr[1].set_ylim([0, 1.0])
            axarr[1].set_xticklabels(xlabels)
            axarr[1].set_xlim([-0.5, self.data.d_ranges[i] + 1 - 0.5])
            plt.savefig('./' + direc + '/f' + str(i) + '.png', bbox_inches='tight')
            plt.close(fig)
        print 'Histograms Generated.'

    def logprob_of_class(self, inputx, class_type, h_alpha, mi=False, mi_f1=5, mi_f2=14, a=2, b=2):
        """
        Generalized form of LogProbability0, LogProbability1 (lp0, lp1)
        :param inputx: input row of data (f_1,...,f_d) to predict on
        :param class_type: bin class 0 or 1? i.e. P or E?
        :param h_alpha: desired hat_alpha
        :param mi: mutual information; if TRUE then execute bonus method with joint dist.
        :param mi_f1: feature1 of joint distribution pair, to determine pair see self.max_mutual_information()
        :param mi_f2: feature2 of joint distribution pair
        :param a: beta bernoulli prior parameter (default:2)
        :param b: beta bernoulli prior parameter (default:2)
        :return: returns LogProbability 0 or 1 depending on parameters
        """

        logprob_of_class = 0.0
        # if non-bonus method, execute normal alg
        if not mi:
            # for each feature category
            for f in range(1, self.data.ndims):
                # calculate multinomial dirilecht for likelihood and add to sum
                k = (self.data.d_ranges[f] + 1)
                numer = np.log(self.train_freq_list[class_type][f][inputx[f]] + h_alpha - 1)
                denom = np.log(self.num_class_list[class_type] + k*h_alpha - k)
                logprob_of_class += numer - denom
            # finally add on beta bernoulli prior
            logprob_of_class += np.log(self.num_class_list[class_type]+a-1) - np.log(self.num_train_data+a+b-2)
        # else bonus method designated from param
        else:
            mi_f_unprocessed = True
            for f in range(1, self.data.ndims):
                # if feature is not one of the joint distribution then calculate normally
                if f != mi_f1 and f != mi_f2:
                    k = (self.data.d_ranges[f] + 1)
                    numer = np.log(self.train_freq_list[class_type][f][inputx[f]] + h_alpha - 1)
                    denom = np.log(self.num_class_list[class_type] + k*h_alpha - k)
                    logprob_of_class += numer - denom
                # else feature is part of joint; treat the joint features as part of a single feature
                else:
                    # disallows double counting
                    if mi_f_unprocessed:
                        # calculates the joint distribution permutation case of (fi,fj)
                        k = (self.data.d_ranges[mi_f1] + 1 + self.data.d_ranges[mi_f2] + 1)
                        numer = np.log(self.train_freq_list[class_type][mi_f1][inputx[mi_f1]] +
                                       self.train_freq_list[class_type][mi_f2][inputx[mi_f2]] + h_alpha - 1)
                        denom = np.log(self.num_class_list[class_type] + k * h_alpha - k)
                        logprob_of_class += numer - denom
                        mi_f_unprocessed = False
            # finally add on beta bernoulli prior for the bonus method
            logprob_of_class += np.log(self.num_class_list[class_type]+a-1) - np.log(self.num_train_data+a+b-2)

        return logprob_of_class

    def predict_data(self, input_data, h_alpha, mi=False, mi_f1=5, mi_f2=14, to_print=True):
        """
        Predicts class based on a row of features.
        :param input_data: 2-D matrix of data cols:(f_1,...,f_d), rows:(1...n)
        :param h_alpha: desired hat alpha for multinomial dirilecht
        :param mi: bonus setting (mutual information) if true then execute bonus method
        :param mi_f1: 1st feature of joint dist.
        :param mi_f2: 2nd feature of joint dist.
        :param to_print: Print Accuracy setting for data set; default TRUE
        :return: returns (accuracy percentage, [list that contains results for each row of data])
        """

        accuracy_cnt = 0.0
        pair_real_pred_list = []
        # for each row of data points
        for i in range(input_data.shape[0]):
            # if regular method
            if not mi:
                lp1 = self.logprob_of_class(input_data[i, :], 1, h_alpha)
                lp0 = self.logprob_of_class(input_data[i, :], 0, h_alpha)
            # else bonus method
            else:
                lp1 = self.logprob_of_class(input_data[i, :], 1, h_alpha, mi, mi_f1, mi_f2)
                lp0 = self.logprob_of_class(input_data[i, :], 0, h_alpha, mi, mi_f1, mi_f2)
            b = max(lp1, lp0)
            # log prob of class 1 given data
            log_prob_1_given_data = lp1 - (np.log(np.exp(lp0 - b) + np.exp(lp1 - b)) + b)

            # if prob of class 1 given data > 0.5 then it is class 1
            label_class = 0
            if log_prob_1_given_data > np.log(0.5):
                label_class = 1

            # if NBC prediction matches actually class increment accuracy counter
            if input_data[i, 0] == label_class:
                accuracy_cnt += 1.0
            # also append row result in list just in case for analysis
            pair_real_pred_list.append((input_data[i, 0], label_class))

        if to_print:
            print "accuracy_rating: " + str(accuracy_cnt/input_data.shape[0])
        return accuracy_cnt/input_data.shape[0], pair_real_pred_list

    def get_accuracy(self, input_data, mi=False, mi_f1=5, mi_f2=14, h_alpha_start=1.0, h_alpha_end=2.0, increment=0.001, plot=True):
        """
        This method executes the NBC on a dataset but within a range of hat alpha values
        It will then generate a line plot of accuracy results based on the variable hat alpha
        :param input_data: 2-D matrix of data cols:(f_1,...,f_d), rows:(1...n)
        :param mi: bonus method if TRUE; default FALSE
        :param mi_f1: feature1 of joint dist.
        :param mi_f2: feature2 of joint dist.
        :param h_alpha_start: start value of range of h_alphas to run on
        :param h_alpha_end: end value of range of h_alphas to run on
        :param increment: how big of a h_alpha step to take for each iteration
        :param plot: to plot or not boolean; default TRUE
        :return: h_alpha values that gave maximum accuracy for training and test data
        """

        # containter lists
        h_alpha_list = []
        accu_train_set = []
        accu_test_set = []
        accu_train_set_mi = []
        accu_test_set_mi = []
        # iterate over range of h_alpha values
        for h_alpha in np.arange(h_alpha_start, h_alpha_end+increment, increment):
            h_alpha_list.append(h_alpha)
            print "step: " + str(h_alpha)
            accu_train_set.append(self.predict_data(input_data.d_train, h_alpha)[0])
            accu_test_set.append(self.predict_data(input_data.d_test, h_alpha)[0])
            # if bonus method wanted do it too
            if mi:
                accu_train_set_mi.append(self.predict_data(input_data.d_train, h_alpha, mi, mi_f1, mi_f2)[0])
                accu_test_set_mi.append(self.predict_data(input_data.d_test, h_alpha, mi, mi_f1, mi_f2)[0])

        # plot line graph
        if plot:
            plt.plot(h_alpha_list, accu_train_set, color='b', label='Train Set')
            plt.plot(h_alpha_list, accu_test_set, color='g', label='Validation Set')
            plt.xlim(1.0, 2.0)
            plt.ylim(0.88, 1.00)
            plt.xlabel('Hat_Alpha')
            plt.ylabel('Accuracy')
            plt.title('Accuracy of NBC for Various Hat_Alpha')
            plt.legend(loc='lower right')
            plt.savefig('./' + 'imgs' + '/accuracy' + '.png', bbox_inches='tight')
            plt.close()
            # plot bonus line graph
            if mi:
                plt.plot(h_alpha_list, accu_train_set, color='b', label='Train Set')
                plt.plot(h_alpha_list, accu_test_set, color='g', label='Validation Set')
                plt.plot(h_alpha_list, accu_train_set_mi, color='c', label='Train Set - 1pair Joint Dist.')
                plt.plot(h_alpha_list, accu_test_set_mi, color='r', label='Validation Set - 1pair Joint Dist.')
                plt.xlim(1.0, 2.0)
                plt.ylim(0.75, 1.00)
                plt.xlabel('Hat_Alpha')
                plt.ylabel('Accuracy')
                plt.title('Accuracy of NBC for Various Hat_Alpha')
                plt.legend(loc='lower right')
                plt.savefig('./' + 'imgs' + '/accuracy_mi' + '.png', bbox_inches='tight')
                plt.close()

        # find h_alpha that gave max accuracy
        max_train, _ = max(enumerate(accu_train_set), key=operator.itemgetter(1))
        max_test, _ = max(enumerate(accu_test_set), key=operator.itemgetter(1))

        return h_alpha_list[max_train], h_alpha_list[max_test]

    def inspect_feature(self, h_alpha):
        """
        Step3: inspect features to see which features have the biggest impact
        :param h_alpha: maximal accuracy h_alpha value
        :return: returns list of list of f_i's sorted by abs() value
        """

        # [ [F_1],...,[F_d] ]
        inspection_list = [[()] * (self.data.d_ranges[f] + 1) for f in range(self.data.ndims)]

        # for each dimension of features
        for i in range(1, self.data.ndims):
            # for each feature dimension's range of possible values
            for j in range(self.data.d_ranges[i]+1):
                # calculate the impact equation
                k = (self.data.d_ranges[i] + 1)
                numer = np.log(self.train_freq_list[1][i][j] + h_alpha - 1)
                denom = np.log(self.num_class_list[1] + k * h_alpha - k)
                lp1 = numer - denom

                numer = np.log(self.train_freq_list[0][i][j] + h_alpha - 1)
                denom = np.log(self.num_class_list[0] + k * h_alpha - k)
                lp0 = numer - denom

                # store result as ('c', impact_value) and sort on abs(impact_value)
                fi = 'f'+str(i)+': '+self.data.reverse_maps[i][j]
                inspection_list[i][j] = (fi, lp1-lp0)
                if j == self.data.d_ranges[i]:
                    inspection_list[i].sort(key=lambda x: abs(x[1]))
                    inspection_list[i].reverse()

        return inspection_list[1:]

    def max_mutual_information(self, h_alpha):
        """
        Calculates the (fi,fj) pair that share the maximum amount of
        mutual information, which can then be used as a joint distribution feature
        :param h_alpha: h_alpha value; ***DO NOT USE 1. DID NOT INCLUDE log(0) HANDLING
                                        ***USE VALUES CLOSE TO 1 i.e. 1.001
        :return: (mi_value, fi, fj); feature pair that shares max mutual information
        """

        mi_list = []
        # for each possible (F_i, F_j) pair
        for i in range(1, self.data.ndims):
            # for each possible (F_i, F_j) pair; i!=j && order matters
            for j in range(i+1, self.data.ndims):
                term = 0.0
                # for each class (0,1)
                for e in range(self.data.d_ranges[0] + 1):
                    # for each possible fi of Fi
                    for fi in range(self.data.d_ranges[i] + 1):
                        # for each possible fj of Fj
                        for fj in range(self.data.d_ranges[j] + 1):
                            # calculate mutual information value
                            k_i = (self.data.d_ranges[i] + 1)
                            k_j = (self.data.d_ranges[j] + 1)
                            k = k_i + k_j

                            p_fifjge_num = (self.train_freq_list[e][i][fi] +
                                            self.train_freq_list[e][j][fj] + h_alpha - 1)
                            p_fifjge_den = (self.num_class_list[e] + k*h_alpha - k)
                            p_fifjge = p_fifjge_num / p_fifjge_den

                            lp_num = np.log(p_fifjge_num) - np.log(p_fifjge_den)
                            numer_fi = np.log(self.train_freq_list[e][i][fi] + h_alpha - 1)
                            denom_fi = np.log(self.num_class_list[e] + k_i * h_alpha - k_i)
                            lp_fi = numer_fi - denom_fi

                            numer_fj = np.log(self.train_freq_list[e][j][fj] + h_alpha - 1)
                            denom_fj = np.log(self.num_class_list[e] + k_j * h_alpha - k_j)
                            lp_fj = numer_fj - denom_fj
                            lp_den = lp_fi + lp_fj
                            lp_term = lp_num - lp_den

                            term += p_fifjge * lp_term
                # store results in list as (mutualinfo_value, i index, j index)
                mi_list.append((term, i, j))

        return max(mi_list)
