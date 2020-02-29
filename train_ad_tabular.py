import numpy as np
from data_loader import Data_Loader
import opt_tc_tabular as tc
import argparse

def load_trans_data(args):
    dl = Data_Loader()
    train_real, val_real, val_fake = dl.get_dataset(args.dataset, args.c_pr)
    y_test_fscore = np.concatenate([np.zeros(len(val_real)), np.ones(len(val_fake))])
    ratio = 100.0 * len(val_real) / (len(val_real) + len(val_fake))

    n_train, n_dims = train_real.shape
    rots = np.random.randn(args.n_rots, n_dims, args.d_out)

    print('Calculating transforms')
    x_train = np.stack([train_real.dot(rot) for rot in rots], 2)
    val_real_xs = np.stack([val_real.dot(rot) for rot in rots], 2)
    val_fake_xs = np.stack([val_fake.dot(rot) for rot in rots], 2)
    x_test = np.concatenate([val_real_xs, val_fake_xs])
    return x_train, x_test, y_test_fscore, ratio


def train_anomaly_detector(args):
    x_train, x_test, y_test, ratio = load_trans_data(args)
    tc_obj = tc.TransClassifierTabular(args)
    f_score = tc_obj.fit_trans_classifier(x_train, x_test, y_test, ratio)
    return f_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--n_rots', default=32, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_epoch', default=25, type=int)
    parser.add_argument('--d_out', default=4, type=int)
    parser.add_argument('--dataset', default='thyroid', type=str)
    parser.add_argument('--exp', default='affine', type=str)
    parser.add_argument('--c_pr', default=0, type=int)
    parser.add_argument('--true_label', default=1, type=int)
    parser.add_argument('--ndf', default=8, type=int)
    parser.add_argument('--m', default=1, type=float)
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--eps', default=0, type=float)
    parser.add_argument('--n_iters', default=500, type=int)

    args = parser.parse_args()
    print("Dataset: ", args.dataset)

    if args.dataset == 'thyroid' or args.dataset == 'arrhythmia':
        n_iters = args.n_iters
        f_scores = np.zeros(n_iters)
        for i in range(n_iters):
            f_scores[i] = train_anomaly_detector(args)
        print("AVG f1_score", f_scores.mean())
    else:
        train_anomaly_detector(args)
