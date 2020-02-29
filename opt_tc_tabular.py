import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import fcnet as model
from sklearn.metrics import precision_recall_fscore_support as prf

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss

def f_score(scores, labels, ratio):
    thresh = np.percentile(scores, ratio)
    y_pred = (scores >= thresh).astype(int)
    y_true = labels.astype(int)
    precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
    return f_score


class TransClassifierTabular():
    def __init__(self, args):
        self.ds = args.dataset
        self.m = args.m
        self.lmbda = args.lmbda
        self.batch_size = args.batch_size
        self.ndf = args.ndf
        self.n_rots = args.n_rots
        self.d_out = args.d_out
        self.eps = args.eps

        self.n_epoch = args.n_epoch
        if args.dataset == "thyroid" or args.dataset == "arrhythmia":
            self.netC = model.netC1(self.d_out, self.ndf, self.n_rots).cuda()
        else:
            self.netC = model.netC5(self.d_out, self.ndf, self.n_rots).cuda()
        model.weights_init(self.netC)
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=args.lr, betas=(0.5, 0.999))


    def fit_trans_classifier(self, train_xs, x_test, y_test, ratio):
        labels = torch.arange(self.n_rots).unsqueeze(0).expand((self.batch_size, self.n_rots)).long().cuda()
        celoss = nn.CrossEntropyLoss()
        print('Training')
        for epoch in range(self.n_epoch):
            self.netC.train()
            rp = np.random.permutation(len(train_xs))
            n_batch = 0
            sum_zs = torch.zeros((self.ndf, self.n_rots)).cuda()

            for i in range(0, len(train_xs), self.batch_size):
                self.netC.zero_grad()
                batch_range = min(self.batch_size, len(train_xs) - i)
                train_labels = labels
                if batch_range == len(train_xs) - i:
                    train_labels = torch.arange(self.n_rots).unsqueeze(0).expand((len(train_xs) - i, self.n_rots)).long().cuda()
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(train_xs[rp[idx]]).float().cuda()
                tc_zs, ce_zs = self.netC(xs)
                sum_zs = sum_zs + tc_zs.mean(0)
                tc_zs = tc_zs.permute(0, 2, 1)

                loss_ce = celoss(ce_zs, train_labels)
                er = self.lmbda * tc_loss(tc_zs, self.m) + loss_ce
                er.backward()
                self.optimizerC.step()
                n_batch += 1

            means = sum_zs.t() / n_batch
            means = means.unsqueeze(0)
            self.netC.eval()

            with torch.no_grad():
                val_probs_rots = np.zeros((len(y_test), self.n_rots))
                for i in range(0, len(x_test), self.batch_size):
                    batch_range = min(self.batch_size, len(x_test) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_test[idx]).float().cuda()
                    zs, fs = self.netC(xs)
                    zs = zs.permute(0, 2, 1)
                    diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)

                    diffs_eps = self.eps * torch.ones_like(diffs)
                    diffs = torch.max(diffs, diffs_eps)
                    logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)

                    val_probs_rots[idx] = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy()

                val_probs_rots = val_probs_rots.sum(1)
                f1_score = f_score(val_probs_rots, y_test, ratio)
                print("Epoch:", epoch, ", fscore: ", f1_score)
        return f1_score

