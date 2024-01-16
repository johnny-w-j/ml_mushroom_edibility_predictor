from data import Data
from nbc_binclass_dirmult import NBCBinClassDirMult

'''
Mushroom procedures to Complete Part 2 and Bonus of 4404 asn1
Author: Woohyuk Jang, wjang@cs.yorku.ca

Mushrooms.csv analyzed with a Naive Bayes Classifer
(Beta Bernoulli Class, Multinomial Dirilecht Features).

See nbc_binclass_dirmult.py for NBC procedures
See data.py for data read and format procedures
'''

# read csv with seed true; set seed false for random permutations of training data
mushrooms = Data('mushrooms.csv', True)
mushrooms.textcat_csv_read()
mushrooms.printstate()

# create NBC instance and train
nbc = NBCBinClassDirMult(mushrooms)
nbc.train_data()
nbc.printstate()

# generate histograms for training data
nbc.generate_hists('Poisonous', 'Edible')

# measure accuracy of NBC on training and test data and return max alpha for accuracy
# USE ALTERNATIVE COMMENTED COMMAND FOR NON-BONUS METHOD
# print nbc.get_accuracy(mushrooms) #non-bonus; runtime=~30mins 64bit ubuntu
print nbc.get_accuracy(mushrooms, True, 5, 14)  # includes bonus; runtime ~1hr 64bit ubuntu
# features 5,14 determined from max_mutual_information() below
print ''

# print accuracy values at max accuracy alpha 1 for record keeping
nbc.predict_data(mushrooms.d_train, 1.0)  # regular method
nbc.predict_data(mushrooms.d_test, 1.0)  # regular method
nbc.predict_data(mushrooms.d_train, 1.0, True, 5, 14)   # bonus method
nbc.predict_data(mushrooms.d_test, 1.0, True, 5, 14)    # bonus method
print ''

# step3 inspect the model: which features have biggest impact on classifier?
for i in range(22):
    print nbc.inspect_feature(1)[i]
    print nbc.inspect_feature(1.001)[i]
    print ''

# bonus procedure: find max mutual information measurement of a pair of features
# result is features 5,14
# to see bonus implementation see nbc.get_accuracy(), nbc.predict_data()
print nbc.max_mutual_information(1.001)
