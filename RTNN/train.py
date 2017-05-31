"""

"""
import sys
from RTNN.model import Model
from six.moves import cPickle
from constant import train_file, dev_file, saved_model, rp_file


def train(sff, epoch, batch_size, step_size, fudge_factor, new=False):
    if new:
        model = Model(5, 1e-6)
        model.preproccess(train_file, dev_file, -1000)
        model.create_rpfile(rp_file.format(i=1), train_file, dev_file)
        model.save(sff)
    else:
        with open(sff, 'rb') as sf:
            model = cPickle.load(sf)
    # model.train_s(train_file + ".p", rp_file, sff, 10, epoch, batch_size, step_size, fudge_factor)
    # model.save(sff)

if __name__ == "__main__":
    sys.exit(train(saved_model.format(i=5), 40, 27, 0.01, 1e-8, True))
