#import tensorflow as tf
import numpy as np
import math


class Normalizer(object):
    @staticmethod
    def l1_normalize(perturb):
        l1n = Normalizer.l1_norm(perturb)
        return perturb / np.maximum(np.reshape(l1n, (-1,) + (1,) * (len(perturb.shape) - 1)), 1e-10)

    @staticmethod
    def l0_normalize(perturb):
        return Normalizer.l1_normalize(perturb)

    @staticmethod
    def linf_normalize(perturb):
        return np.sign(perturb)

    @staticmethod
    def l0_norm(perturb):
        return (np.abs(np.reshape(perturb, (perturb.shape[0], -1))) != 0).sum(axis=1).float()

    @staticmethod
    def l1_norm(perturb):
        return np.abs(np.reshape(perturb, (perturb.shape[0], -1))).sum(axis=1)

    @staticmethod
    def l2_norm(perturb):
        size = np.prod(perturb.shape[1:]).astype(np.int64)
        norm = np.linalg.norm(np.reshape(perturb, [-1, size]), 2, axis=1)
        return norm

    @staticmethod
    def l2_normalize(perturb):
        l2n = np.maximum(Normalizer.l2_norm(perturb), 0.0001)
        perturb = perturb / \
            np.reshape(l2n, (-1, ) + (1,) * (len(perturb.shape) - 1))
        return perturb

    @staticmethod
    def normalize(perturb, p):
        normalizers = {"l0": Normalizer.l0_normalize, "l1": Normalizer.l1_normalize,
                       "l2": Normalizer.l2_normalize, "linf": Normalizer.linf_normalize}
        return normalizers[p](perturb)


class Bounder(object):
    @staticmethod
    def linf_bound(perturb, epsilon):
        return np.clip(perturb, a_min=-epsilon, a_max=epsilon)

    @staticmethod
    def l0_bound(perturb, epsilon):
        reshaped_perturb = np.reshape(perturb, (perturb.shape[0], -1))
        sorted_perturb = np.sort(reshaped_perturb, axis=1)
        k = int(math.ceil(epsilon))
        thresholds = sorted_perturb[:, -k]
        mask = perturb >= np.reshape(
            thresholds, (perturb.shape[0], ) + (1,) * (len(perturb.shape) - len(thresholds.shape)))
        return perturb * mask

    @staticmethod
    def l1_bound(perturb, epsilon):
        bounded_s = []
        for i in range(perturb.shape[0]):
            bs = perturb[i]
            abs_bs = np.abs(bs)
            if np.sum(abs_bs) > epsilon:
                old_shape = bs.shape
                bs = Bounder.projection_simplex_sort(
                    np.reshape(abs_bs, (abs_bs.size, )), epsilon)
                bs = np.reshape(bs, old_shape)
            bounded_s.append(bs)
        return np.array(bounded_s)

    @staticmethod
    def projection_simplex_sort(v, z=1):
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    @staticmethod
    def l2_bound(perturb, epsilon):
        l2_norm = np.expand_dims(Normalizer.l2_norm(perturb), 1)
        multiplier = 1.0 / np.maximum(l2_norm / epsilon, np.ones_like(l2_norm))
        return perturb * np.reshape(multiplier, (-1, ) + (1,) * (len(perturb.shape) - 1))

    @staticmethod
    def bound(perturb, epsilon, p):
        bounders = {"l0": Bounder.l0_bound, "l1": Bounder.l1_bound,
                    "l2": Bounder.l2_bound, "linf": Bounder.linf_bound}
        return bounders[p](perturb, epsilon)
