"""

"""
import sys
from RTNN.model import Model
from constant import train_file, dev_file, saved_model, rp_file


def train(max, epoch, batch_size):
    for i in range(max):
        model = Model(5, 1e-6)
        model.train(train_file, dev_file, rp_file.format(i=i), saved_model.format(i=i), 10, -3000, epoch, batch_size)
        model.save(saved_model.format(i=i))

if __name__ == "__main__":
    sys.exit(train(5, 40, 27))
