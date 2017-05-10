"""
This file is used to test trained model.
"""
# import sys
from six.moves import cPickle

from constant import saved_model, train_file, test_file, test_rp

with open(saved_model, 'rb') as f:
    model = cPickle.load(f)

print(model.test(train_file, test_rp))
