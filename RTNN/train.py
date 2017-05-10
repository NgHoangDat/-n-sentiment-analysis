"""

"""
import sys
from RTNN.model import Model
from six.moves import cPickle
from constant import train_file, proccessed_file, saved_model, rp_file


def train(epoch, batch_size, step_size, fudge_factor, new=False):
    if new:
        model = Model(5, 1e-6)
        model.preproccess(train_file, proccessed_file, -3000)
        model.save(saved_model)
    else:
        with open(saved_model, 'rb') as sf:
            model = cPickle.load(sf)
    model.train(proccessed_file, rp_file, saved_model, 10, epoch, batch_size, step_size, fudge_factor)
    model.save(saved_model)

if __name__ == "__main__":
    sys.exit(train(1000, 27, 0.01, 1e-8, True))
