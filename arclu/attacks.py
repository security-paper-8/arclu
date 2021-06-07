import sys
sys.path.append("../ar_comparison")
import torch
import eval_codes.attacks.attack_torch as attack_torch
import eval_codes.attacks.attack_common as attack_common
import numpy as np


class PGDOursAttack(attack_torch.PGDAttack):
    def get_loss(self, x, x_adv, y_victim, y_target, device="cuda:0"):
        o = self.model(x_adv)
        dist = self.model.compute_squared_distance(o)
        num_classes = dist.size(1)
        if not (y_target is None):
            target_one_hot = torch.eye(num_classes)[y_target].to(device)
            other_one_hot = 1.0 - target_one_hot
            target_dist = torch.sum(dist * target_one_hot, dim=1)
            loss = -target_dist.mean()
        else:
            true_one_hot = torch.eye(num_classes)[y_victim].to(device)
            other_one_hot = 1.0 - true_one_hot
            other_dist, other_pred = torch.min(
                dist * other_one_hot + true_one_hot * 999999, dim=1)
            loss = -other_dist.mean()
        return loss


class PGDDisperseOursAttack(attack_torch.PGDAttack):
    def __init__(self, model, M, n_latent, n_class, zero_threshold_multiplier, adv_coeff):
        super(PGDDisperseOursAttack, self).__init__(model)
        self.M = M
        self.n_latent = n_latent
        self.n_class = n_class
        self.zero_threshold_multiplier = zero_threshold_multiplier
        self.adv_coeff = adv_coeff

    def get_loss(self, x, x_adv, y_victim, y_target, device="cuda:0"):
        o = self.model(x_adv)
        dist = self.model.compute_squared_distance(o)
        num_classes = dist.size(1)
        if not (y_target is None):
            target_one_hot = torch.eye(num_classes)[y_target].to(device)
            other_one_hot = 1.0 - target_one_hot
            target_dist = torch.sum(dist * target_one_hot, dim=1)
            loss = -target_dist.mean()
        else:
            true_one_hot = torch.eye(num_classes)[y_victim].to(device)
            other_one_hot = 1.0 - true_one_hot
            other_dist, other_pred = torch.min(
                dist * other_one_hot + true_one_hot * 999999, dim=1)
            loss = -other_dist.mean()

        squared_dist_from_adv_c = torch.pow(o - 0.0, 2).sum(dim=1)
        M = self.M
        n_latent = self.n_latent
        n_class = self.n_class
        dist_adv_c_threshold = M * M * (n_latent // n_class)

        zero_threshold_multiplier = self.zero_threshold_multiplier
        coeff = self.adv_coeff
        squared_dist_from_adv_c[squared_dist_from_adv_c >=
                                dist_adv_c_threshold * zero_threshold_multiplier] = 0
        return loss + coeff * squared_dist_from_adv_c.mean()


class NESOursAttack(attack_torch.NESAttack):
    def __init__(self, model: torch.nn.Module, n_samples: int, search_sigma: float = 0.01):
        super(NESOursAttack, self).__init__(model, n_samples, search_sigma)

    def compute_nes_target_value(self, x_victim, x_adv, y_victim, y_target):
        logits = self.model(x_adv)
        dist_list = self.model.compute_squared_distance(logits)
        true_mask = torch.eye(dist_list.size(1), device="cuda:0")[y_victim]
        target_dist = torch.min(dist_list + true_mask * 99999999, dim=1)[0]
        print(target_dist.mean(), target_dist.size())
        return -target_dist


class BoundaryOursAttack(attack_torch.BoundaryAttack):
    def __init__(self, model: torch.nn.Module, n_delta: int, thresholds_each_class):
        super(BoundaryOursAttack, self).__init__(model, n_delta)
        self.thresholds_each_class = torch.from_numpy(
            thresholds_each_class).cuda()

    def is_success(self, x, y_target):
        logits = self.model(x)
        dist_list = self.model.compute_squared_distance(logits)
        min_dist, pred = dist_list.min(dim=1)

        mislead_success = (pred != y_target)
        bypass_success = (min_dist < self.thresholds_each_class[pred])

        return bypass_success & mislead_success


class TransferOursAttack(attack_common.TransferAttack):
    def __init__(self, dataset, model, thresholds_each_class):
        super().__init__(dataset)
        self.model = model
        self.m_thresholds_each_class = torch.from_numpy(
            thresholds_each_class).cuda()

    def is_benign_okay(self, x_benign, y_true):
        x_benign = torch.from_numpy(x_benign).cuda()
        y_true = torch.from_numpy(y_true).cuda()

        logits = self.model(x_benign)
        dist_list = self.model.compute_squared_distance(logits)
        min_dist, pred = dist_list.min(dim=1)

        detection_okay = (min_dist < self.m_thresholds_each_class[pred])

        pred_okay = (pred == y_true)
        return (pred_okay & detection_okay).cpu().detach().view(-1, 1).squeeze(1).numpy()


class RepresentationOursAttack(attack_torch.PGDAttack):
    def get_loss(self, x, x_adv, y_victim, y_target, device="cuda:0"):
        o = self.model(x_adv)
        assert(not(y_target is None))
        loss = -torch.pow(o.unsqueeze(1) - y_target.cuda(),
                          2).sum(dim=2).mean(dim=1).mean(dim=0)
        return loss


class ModelForFoolbox(torch.nn.Module):
    def __init__(self, model):
        super(ModelForFoolbox, self).__init__()
        self.model = model

    def forward(self, x):
        return -self.model.compute_squared_distance(self.model(x))


class ModelForRescale(torch.nn.Module):
    def __init__(self, model, min, max):
        super(ModelForRescale, self).__init__()
        self.model = model
        self.min = min
        self.max = max

    def forward(self, x):
        rescaled_x = x * (self.max - self.min) + self.min
        return -self.model.compute_squared_distance(self.model(rescaled_x))


class PGDOneOursAttack(attack_torch.PGDAttack):
    def get_loss(self, x, x_adv, y_victim, y_target, device="cuda:0"):
        o = self.model(x_adv)
        dist = self.model.compute_squared_distance(o)
        num_classes = dist.size(1)
        if not (y_target is None):
            loss = - \
                (self.model.y_mu_encoding[y_target] -
                 o).abs().max(dim=1)[0].mean()
        else:
            true_one_hot = torch.eye(num_classes)[y_victim].to(device)
            other_one_hot = 1.0 - true_one_hot
            other_dist, other_pred = torch.min(
                dist * other_one_hot + true_one_hot * 999999, dim=1)

            loss = - \
                (self.model.y_mu_encoding[other_pred] -
                 o).abs().max(dim=1)[0].mean()

        return loss
