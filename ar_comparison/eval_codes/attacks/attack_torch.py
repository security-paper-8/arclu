import os
import sys
sys.path.append(os.path.dirname(__file__) + "/../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.torch import Normalizer, Bounder


# Template classes
class Attack(object):
    def __init__(self, model):
        self.model = model

    def get_loss(self, x, perturb, y_victim, y_target):
        raise NotImplementedError()

    def compute_grad(self, x_victim, x_adv, y_victim, y_target):
        loss = self.get_loss(x_victim, x_adv, y_victim, y_target)
        loss.backward()
        return x_adv.grad.data

    def perturb(self, x_victim, y_victim, y_target, p, epsilon, step_size, max_iters, min_value, max_value, decay=0.00, device="cuda:0", random_perturb_start=False, random_epsilon=None, repetition=1, multi_targeted=False, verbose=True, iter_test=False):
        if multi_targeted is True and verbose == True:
            print("*** `multi_targeted` is set to True")
            print("*** `repetition` should be  `num_classes`.")
            print(
                "*** `y_target` is ignored but shouldn't be None. (`targeted` should be set)")
            assert(not (y_target is None))
        if iter_test == True:
            assert(multi_targeted == False)

        self.model.eval()
        x_adv_repeat_list = []
        x_adv_iters = []
        x_victim = x_victim.to(device)
        y_victim = y_victim.to(device)
        self.model.to(device)
        for r in range(repetition):
            x_adv = x_victim.clone().to(device)
            if multi_targeted is True:
                y_target = r * torch.ones_like(y_target)

            if random_perturb_start:
                noise = torch.rand(x_adv.size()).to(device)
                normalized_noise = Normalizer.normalize(noise, p)
                if random_epsilon == None:
                    x_adv += normalized_noise * epsilon * 0.1
                else:
                    x_adv += normalized_noise * random_epsilon

            momentum = torch.zeros_like(x_adv)
            self.model.eval()
            for i in range(max_iters):
                if iter_test == True and i % 10 == 0:
                    x_adv_iters.append(x_adv.cpu().detach())
                x_adv.requires_grad = True
                self.model.zero_grad()
                grad = self.compute_grad(x_victim, x_adv, y_victim, y_target)
                momentum = grad + momentum * decay
                x_adv = (x_adv + Normalizer.normalize(momentum, p) * step_size)
                if p != "l0":
                    perturb = Bounder.bound(x_adv - x_victim, epsilon, p)
                    x_adv = torch.clamp(x_victim + perturb,
                                        min=min_value, max=max_value).detach()
                else:
                    perturb = Bounder.l0_bound_sparse(
                        x_adv - x_victim, epsilon, x_victim)
                    x_adv = (x_victim + perturb).detach()
            x_adv_repeat_list.append(x_adv.unsqueeze(0))

        x_adv_repeat = torch.cat(x_adv_repeat_list, dim=0).transpose(
            0, 1).reshape((repetition * x_victim.size(0),) + x_victim.size()[1:])

        if iter_test == False:
            return x_adv_repeat  # x_adv
        else:
            return x_adv_iters


class BoundaryAttack(object):
    def __init__(self, model, n_delta):
        self.model = model
        self.n_delta = n_delta

    def is_success(self, x, y_target):
        return self.model(x).max(dim=1)[1] != y_target

    def get_init_noise(self, x_target, y_target, min_value, max_value):
        x_init = (max_value - min_value) * \
            torch.rand(x_target.size()).cuda() + min_value
        x_init = torch.clamp(x_init, min=min_value, max=max_value)
        for i in range(x_target.size(0)):
            N_TRY = 20
            for r in range(N_TRY):
                if self.is_success(x_init[i].unsqueeze(0), y_target[i].unsqueeze(0)).cpu().numpy() == True:
                    print("Success getting init noise", i)
                    break
                else:
                    x_init[i] = (max_value - min_value) * \
                        torch.rand(x_target[0].size()) + min_value
                    x_init[i] = torch.clamp(
                        x_init[i], min=min_value, max=max_value)
                if r == N_TRY - 1:
                    print("Failed getting init noise", i)

        return x_init

    def perturb(self, x_target, y_target, p, perturb_size, init_delta, init_epsilon, max_iters, min_value, max_value, device="cuda:0", requires_grad=False):
        x_target = x_target.to(device)
        y_target = y_target.to(device)
        self.model.to(device)
        self.model.eval()
        x_init = self.get_init_noise(x_target, y_target, min_value, max_value)
        x_adv = x_init.clone()
        delta = torch.ones((x_target.size(0), 1),
                           device=device).float() * init_delta
        epsilon = torch.ones((x_target.size(0), 1),
                             device=device).float() * init_epsilon

        size_with_noise = [x_adv.size(0), self.n_delta] + list(x_adv.size())[1:]
        squeezed_size_with_noise = [x_adv.size(
            0) * self.n_delta] + list(x_adv.size())[1:]

        x_init_repeat = x_init.repeat(
            1, self.n_delta, 1, 1).view(*squeezed_size_with_noise)
        x_target_repeat = x_target.repeat(
            1, self.n_delta, 1, 1).view(*squeezed_size_with_noise)
        y_target_repeat = y_target.view(-1, 1).repeat(1,
                                                      self.n_delta).view(-1, 1).squeeze(1)

        class dummy_context_mgr():
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        nograd_context = torch.no_grad()
        null_context = dummy_context_mgr()
        cm = nograd_context if requires_grad == False else null_context
        with cm:
            for i in range(max_iters):
                # step 1
                x_adv_repeat = x_adv.repeat(1, self.n_delta, 1, 1).view(
                    *squeezed_size_with_noise)
                delta_repeat = delta.repeat(
                    1, self.n_delta).view(-1, 1).squeeze(1)

                # [n_adv*n_delta, n_channel, n_height, n_width]
                step1_noise = torch.randn(
                    squeezed_size_with_noise, device=device)
                target_distance = Normalizer.l2_norm(
                    x_adv - x_target).view(-1, 1).repeat(1, self.n_delta).view(-1, 1).squeeze(1)

                bounded_step1_noise = Bounder.bound(
                    step1_noise, target_distance * delta_repeat, "l2")
                bounded_step1_noise_added_projected = Normalizer.normalize(
                    x_adv_repeat + bounded_step1_noise - x_target_repeat, "l2") * target_distance.view(squeezed_size_with_noise[0], 1, 1, 1) + x_target_repeat
                bounded_step1_noise_added_projected = torch.clamp(
                    bounded_step1_noise_added_projected, max=max_value, min=min_value)
                step1_success = self.is_success(
                    bounded_step1_noise_added_projected, y_target_repeat)

                step1_success_folded = step1_success.view(
                    x_adv.size(0), self.n_delta)
                step1_success_ratio = step1_success_folded.float().mean(dim=1)

                bounded_step1_noise_added_projected_selected = []
                for j in range(x_target.size(0)):
                    for k in range(self.n_delta):
                        if step1_success_folded[j][k] == True:
                            bounded_step1_noise_added_projected_selected.append(
                                bounded_step1_noise_added_projected[j * self.n_delta + k].unsqueeze(0))
                            break
                        elif k == self.n_delta - 1:
                            bounded_step1_noise_added_projected_selected.append(
                                bounded_step1_noise_added_projected[j * self.n_delta].unsqueeze(0))

                bounded_step1_noise_added_projected_selected = torch.cat(
                    bounded_step1_noise_added_projected_selected, dim=0)
                # step 2
                approaching_to_target = bounded_step1_noise_added_projected_selected - \
                    epsilon.view(x_adv.size(0), 1, 1, 1) * \
                    (bounded_step1_noise_added_projected_selected - x_target)

                step2_success = self.is_success(approaching_to_target, y_target)
                all_success = ((step1_success_ratio > 0) & (
                    step2_success == True)).view(-1, 1, 1, 1).float()

                x_adv = x_adv * (1.0 - all_success) + \
                    approaching_to_target * all_success

                # delta update
                delta = torch.clamp(1.1 * delta * (step1_success_ratio.unsqueeze(1) > 0.5).float(
                ) + 0.9 * delta * (step1_success_ratio.unsqueeze(1) <= 0.5).float(), min=0.01, max=1.0)

                # epsilon update
                epsilon = torch.clamp(1.1 * epsilon * (step2_success.float()).unsqueeze(
                    1) + 0.9 * epsilon * (1.0 - step2_success.float()).unsqueeze(1), min=0.0, max=0.99)

        # Filter only examples with small perturbations
        perturb = (x_adv - x_target)
        bounded_perturb = Bounder.bound(perturb, perturb_size, p)
        is_perturb_small_enough = ((perturb - bounded_perturb).reshape(-1, np.prod(
            x_adv.size()[1:])).abs().sum(dim=1) == 0).float().view(-1, 1, 1, 1)
        x_adv = x_adv * (is_perturb_small_enough) + x_target * \
            (1.0 - is_perturb_small_enough)

        return x_adv


# Real Implementation

class PGDAttack(Attack):
    def __init__(self, model):
        super(PGDAttack, self).__init__(model)

    def get_loss(self, x, x_adv, y_victim, y_target, device="cuda:0"):

        logits = self.model(x_adv)
        num_classes = logits.size(1)
        if y_target is not None:
            target_one_hot = torch.eye(num_classes)[y_target].to(device)
            other_one_hot = 1.0 - target_one_hot
            target_logit = torch.sum(logits * target_one_hot, dim=1)
            other_logit = torch.max(
                logits * other_one_hot - target_one_hot * 999999, dim=1)[0]
            diff = torch.nn.functional.relu(other_logit - target_logit + 10)
            loss = -torch.mean(diff)
        else:
            true_one_hot = torch.eye(num_classes)[y_victim].to(device)
            other_one_hot = 1.0 - true_one_hot
            true_logit = torch.sum(logits * true_one_hot, dim=1)
            other_logit = torch.max(
                logits * other_one_hot - true_one_hot * 999999, dim=1)[0]
            diff = torch.nn.functional.relu(true_logit - other_logit + 10)
            loss = -torch.mean(diff)

        return loss


class NESAttack(Attack):
    def __init__(self, model, n_samples, search_sigma=0.01):
        super(NESAttack, self).__init__(model)
        self.search_sigma = search_sigma
        self.n_samples = n_samples

    def get_noise_added_x(self, x_adv):
        sampling = torch.randn((x_adv.size(0), self.n_samples, x_adv.size(
            1), x_adv.size(2), x_adv.size(3))).cuda()
        noise_added_x_adv = torch.unsqueeze(
            x_adv, 1) + self.search_sigma * sampling
        return sampling, noise_added_x_adv

    def compute_nes_target_value(self, x_victim, x_adv, y_victim, y_target):
        logits = self.model(x_adv)

        true_mask = torch.eye(logits.size(1), device="cuda:0")[y_victim]
        true_logit = torch.sum(logits * true_mask, dim=1)
        target_logit = torch.max(logits - 999999.0 * true_mask, dim=1)[0]
        print(target_logit.mean())
        return (target_logit - true_logit)

    # It does not call get_loss()
    def compute_grad(self, x_victim, x_adv, y_victim, y_target):
        sampling, noise_added_x_adv = self.get_noise_added_x(x_adv)
        y_victim_repeat = y_victim.view(-1, 1).repeat(1,
                                                      self.n_samples).view(-1, 1).squeeze(1)
        target_nes_value_list = []
        noise_added_x_adv_flatten = noise_added_x_adv.view(
            -1, x_adv.size(1), x_adv.size(2), x_adv.size(3))

        noise_batch_size = 128
        for i in range(0, noise_added_x_adv_flatten.size(0), noise_batch_size):
            batch_start = i
            batch_end = min(noise_added_x_adv_flatten.size(0),
                            batch_start + noise_batch_size)
            target_nes_value_list.append(self.compute_nes_target_value(None, noise_added_x_adv_flatten[batch_start:batch_end],
                                                                       y_victim_repeat[batch_start:batch_end],
                                                                       None))
        target_nes_value = torch.cat(target_nes_value_list, dim=0).view(
            x_adv.size(0), self.n_samples)
        mean, std = torch.mean(target_nes_value, dim=1, keepdim=True), torch.std(
            target_nes_value, dim=1, keepdim=True)
        normalized_nes_logit = (target_nes_value - mean) / std

        normalized_f = True
        if normalized_f:
            e_grad = torch.mean(normalized_nes_logit.view(x_adv.size(
                0), self.n_samples, 1, 1, 1) * sampling, dim=1) / self.search_sigma
        else:
            e_grad = torch.mean(target_nes_value.view(x_adv.size(
                0), self.n_samples, 1, 1, 1) * sampling, dim=1) / self.search_sigma
        return e_grad

    def perturb(self, x_victim, y_victim, y_target, p, epsilon, step_size, max_iters, min_value, max_value, device="cuda:0"):
        with torch.no_grad():
            return super(NESAttack, self).perturb(x_victim, y_victim, y_target, p, epsilon, step_size, max_iters, min_value, max_value, device=device)
