from io import BytesIO
import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def TSNE_transform(x):
    model = TSNE(2)
    x = model.fit_transform(x)
    return x


def feat2emb(feats, labels, transform=None):

    if feats.ndim != 2 or feats.shape[1] < 2:
        raise ValueError('feats should has shape [N, C], C >= 2.')

    labels_set = set()
    for l in labels:
        labels_set.add(l)

    if transform:
        feats = transform(feats)

    fig = plt.gcf()
    fig.clear()
    for l in labels_set:
        c_feats = feats[labels == l]
        plt.scatter(c_feats[:, 0], c_feats[:, 1], label=l)
    plt.legend()

    tmp_io = BytesIO()
    fig.savefig(tmp_io, format='jpg')
    return tmp_io.getvalue()


class TimeMeter:

    def __init__(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.duration += time.perf_counter() - self.start_time
        self.counter += 1

    def get(self):
        return self.duration / self.counter

    def reset(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.

    def get_and_reset(self):
        val = self.get()
        self.reset()
        return val