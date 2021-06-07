import sys
import os
sys.path.append(os.path.dirname(__file__) + "/../../")
import tensorflow as tf
import numpy as np
from tqdm import tqdm
print(os.path.dirname(__file__) + "../../")
from eval_codes.utils.np import Normalizer, Bounder
from eval_codes.utils.tf import batch_wise_sess_run


class Attack(object):
    def __init__(self, model, X_in, Y_in, Y_target_in, num_classes):
        self.model = model
        self.X_in = X_in
        self.Y_in = Y_in
        self.Y_target_in = Y_target_in
        self.num_classes = num_classes
        self.set_loss_and_grad_graph()
        print("After set loss and grad graph")
        assert(self.grad != None)
        assert(self.loss != None)

    def set_loss_and_grad_graph(self):
        raise NotImplementedError()

    def compute_grad(self, x_adv, y_victim, y_target, sess, x):
        raise NotImplementedError()

    def perturb(self, x_victim, y_victim, y_target, p, epsilon, step_size, max_iters, min_value, max_value,
                sess, random_perturb_start=False, random_epsilon=None, repetition=1, multi_targeted=False):
        if multi_targeted is True:
            print("*** `multi_targeted` is set to True")
            print("*** `repetition` should be  `num_classes`.")
            print(
                "*** `y_target` is ignored but shouldn't be None. (`targeted` should be set)")
            assert(not (y_target is None))
        print("random start:", random_perturb_start)
        x_adv_repeat_list = []
        for r in range(repetition):
            if multi_targeted is True:
                y_target = r * np.ones_like(y_target)

            x_adv = x_victim.copy()
            if random_perturb_start:
                noise = np.random.uniform(size=x_adv.shape)
                normalized_noise = Normalizer.normalize(noise, p)
                if random_epsilon == None:
                    x_adv += normalized_noise * epsilon * 0.1
                else:
                    x_adv += normalized_noise * random_epsilon

            for i in range(max_iters):
                grad = self.compute_grad(
                    x_adv, y_victim, y_target, sess, x_victim)

                x_adv = (x_adv + Normalizer.normalize(grad, p) * step_size)
                perturb = Bounder.bound(x_adv - x_victim, epsilon, p)
                x_adv = np.clip(x_victim + perturb,
                                a_min=min_value, a_max=max_value)
            x_adv_repeat_list.append(np.expand_dims(x_adv, 0))
        print(np.concatenate(x_adv_repeat_list, axis=0).swapaxes(0, 1).shape)
        x_adv_repeat = np.concatenate(x_adv_repeat_list, axis=0).swapaxes(
            0, 1).reshape((repetition * x_victim.shape[0],) + x_victim.shape[1:])
        return x_adv_repeat


class BoundaryAttack(object):
    def __init__(self, model, X_in, Y_in, Y_target_in, num_classes, num_delta):
        self.model = model
        self.X_in = X_in
        self.Y_in = Y_in
        self.Y_target_in = Y_target_in
        self.num_classes = num_classes
        self.num_delta = num_delta
        self.set_graph()

    def set_graph(self):
        self.output = self.model(self.X_in)
        self.pred = tf.argmax(self.output, axis=1)

    def is_success(self, x_adv, y_victim, sess, x):
        p = sess.run(self.pred, feed_dict={self.X_in: x_adv})
        return p != y_victim

    def get_init_noise(self, x_victim, y_victim, min_value, max_value, sess):
        x_init = (max_value - min_value) * \
            np.random.rand(*x_victim.shape) + min_value
        NUM_TRY = 20
        for i in range(x_victim.shape[0]):
            is_fail = False
            for nt in range(NUM_TRY):
                if self.is_success(np.expand_dims(x_init[i], 0), np.expand_dims(y_victim[i], 0), sess, x_victim) == True:
                    print("Success getting init noise: {}".format(i))
                    break
                else:
                    x_init[i] = (max_value - min_value) * \
                        np.random.rand(*x_victim[0].shape) + min_value
                    if nt == NUM_TRY - 1:
                        is_fail = True
            if is_fail:
                print("Failed getting init noise", i)
        return x_init

    def perturb(self, x_victim, y_victim, p, perturb_size, init_delta, init_epsilon, max_iters, min_value, max_value, sess):
        x_init = self.get_init_noise(
            x_victim, y_victim, min_value, max_value, sess)
        x_adv = np.copy(x_init)
        delta = np.ones((x_victim.shape[0], 1)) * init_delta
        epsilon = np.ones((x_victim.shape[0], 1)) * init_epsilon

        size_with_noise = [x_victim.shape[0],
                           self.num_delta] + list(x_victim.shape[1:])
        squeezed_size_with_noise = [x_victim.shape[0]
                                    * self.num_delta] + list(x_victim.shape[1:])

        x_init_repeat = np.tile(
            x_init, (1, self.num_delta) + (1,) * (len(x_victim.shape) - 2))
        x_victim_repeat = np.reshape(np.tile(
            x_victim, (1, self.num_delta) + (1,) * (len(x_victim.shape) - 2)), squeezed_size_with_noise)

        y_victim_repeat = np.squeeze(np.reshape(np.tile(np.expand_dims(
            y_victim, 1), (1, self.num_delta)), (squeezed_size_with_noise[0], 1)), 1)

        for i in range(max_iters):
            x_adv_repeat = np.reshape(np.tile(
                x_adv, (1, self.num_delta) + (1,) * (len(x_victim.shape) - 2)), squeezed_size_with_noise)
            delta_repeat = np.squeeze(np.reshape(
                np.tile(delta, (1, self.num_delta)), (squeezed_size_with_noise[0], 1)), 1)

            step1_noise = np.random.rand(*squeezed_size_with_noise)
            target_distance = np.squeeze(
                np.reshape(
                    np.tile(np.expand_dims(Normalizer.l2_norm(
                        x_adv - x_victim), 1), (1, self.num_delta)),
                    (squeezed_size_with_noise[0], 1)),
                1)

            bounded_step1_noise = Bounder.bound(
                step1_noise, np.expand_dims(target_distance * delta_repeat, 1), "l2")
            bounded_step1_noise_added_projected = Normalizer.normalize(x_adv_repeat + bounded_step1_noise - x_victim_repeat, "l2") * np.reshape(
                target_distance, (squeezed_size_with_noise[0],) + (1,) * (len(x_victim.shape) - 1)) + x_victim_repeat
            bounded_step1_noise_added_projected = np.clip(
                bounded_step1_noise_added_projected, a_min=min_value, a_max=max_value)
            step1_success = self.is_success(
                bounded_step1_noise_added_projected, y_victim_repeat, sess, x_victim_repeat)
            step1_success_folded = np.reshape(
                step1_success, (x_victim.shape[0], self.num_delta))
            step1_success_ratio = np.mean(
                step1_success_folded.astype(np.float32), axis=1)

            bounded_step1_noise_added_projected_selected = []
            for j in range(x_victim.shape[0]):
                for k in range(self.num_delta):
                    if step1_success_folded[j][k] == True:
                        bounded_step1_noise_added_projected_selected.append(np.expand_dims(
                            bounded_step1_noise_added_projected[j * self.num_delta + k], 0))
                        break
                    elif k == self.num_delta - 1:
                        bounded_step1_noise_added_projected_selected.append(np.expand_dims(
                            bounded_step1_noise_added_projected[j * self.num_delta], 0))

            bounded_step1_noise_added_projected_selected = np.concatenate(
                bounded_step1_noise_added_projected_selected, axis=0)
            # step 2
            approaching_to_target = bounded_step1_noise_added_projected_selected - np.reshape(epsilon, (x_victim.shape[0],) + (
                1,) * (len(x_victim.shape) - 1)) * (bounded_step1_noise_added_projected_selected - x_victim)

            step2_success = self.is_success(
                approaching_to_target, y_victim, sess, x_victim_repeat)
            all_success = np.reshape(((step1_success_ratio > 0) & (
                step2_success == True)), (x_victim.shape[0],) + (1,) * (len(x_victim.shape) - 1))

            x_adv = x_adv * (~all_success) + approaching_to_target * all_success

            # delta update
            delta = np.clip(1.1 * delta * (np.expand_dims(step1_success_ratio, 1) > 0.5) + 0.9 *
                            delta * (np.expand_dims(step1_success_ratio, 1) <= 0.5), a_min=0.02, a_max=1.0)

            # epsilon update
            epsilon = np.clip(1.1 * epsilon * np.expand_dims(step2_success, 1) + 0.9 *
                              epsilon * np.expand_dims(~step2_success, 1), a_min=0.0, a_max=0.99)

        # Filter only examples with small perturbations
        perturb = (x_adv - x_victim)
        bounded_perturb = Bounder.bound(perturb, perturb_size, p)
        is_perturb_small_enough = (np.abs((perturb - bounded_perturb).reshape((perturb.shape[0], -1))).sum(
            axis=1) == 0).astype(np.float32).reshape((-1,) + (1,) * (len(x_victim.shape) - 1))

        x_adv = x_adv * (is_perturb_small_enough) + x_victim * \
            (1.0 - is_perturb_small_enough)

        return x_adv


class PGDAttack(Attack):
    def __init__(self, model, X_in, Y_in, Y_target_in, num_classes):
        super(PGDAttack, self).__init__(
            model, X_in, Y_in, Y_target_in, num_classes)

    def set_loss_and_grad_graph(self):
        self.output = self.model(self.X_in)
        # Untargeted attack
        if self.Y_target_in == None:
            one_hot_encoding = tf.one_hot(self.Y_in, self.num_classes)
            other_hot_encoding = 1.0 - one_hot_encoding

            true_logit = tf.reduce_sum(self.output * one_hot_encoding, axis=1)
            other_logit = tf.reduce_max(
                self.output * other_hot_encoding - one_hot_encoding * 99999.0, axis=1)

            self.loss = -tf.nn.relu(true_logit - other_logit + 10)
        else:
            one_hot_encoding = tf.one_hot(self.Y_target_in, self.num_classes)
            other_hot_encoding = 1.0 - one_hot_encoding
            target_logit = tf.reduce_sum(self.output * one_hot_encoding, axis=1)
            other_logit = tf.reduce_max(
                self.output * other_hot_encoding - one_hot_encoding * 99999.0, axis=1)

            self.loss = -tf.nn.relu(other_logit - target_logit + 10)

        self.grad = tf.gradients(self.loss, [self.X_in])[0]

    def compute_grad(self, x_adv, y_victim, y_target, sess, x):
        fd = None
        if self.Y_target_in == None:
            fd = {self.X_in: x_adv, self.Y_in: y_victim}
        else:
            fd = {self.X_in: x_adv, self.Y_in: y_victim,
                  self.Y_target_in: y_target}
        grad = sess.run(self.grad, feed_dict=fd)
        return grad


class NESAttack(Attack):
    def __init__(self, model, X_in, Y_in, Y_target_in, num_classes, n_samples, search_sigma=0.01):
        self.n_samples = n_samples
        self.search_sigma = search_sigma
        assert(Y_target_in != None)
        super(NESAttack, self).__init__(
            model, X_in, Y_in, Y_target_in, num_classes)

    def set_loss_and_grad_graph(self):
        input_shape = self.X_in.get_shape().as_list()
        noise_shape = [tf.shape(self.X_in)[0], self.n_samples] + input_shape[1:]

        random_noise = tf.random.normal(noise_shape)
        noise_added_x_adv = tf.expand_dims(
            self.X_in, 1) + random_noise * self.search_sigma
        output = tf.reshape(self.model(noise_added_x_adv),
                            [-1, self.n_samples, self.num_classes])

        target_mask = tf.one_hot(self.Y_target_in, self.num_classes)
        target_logit = tf.reduce_sum(
            output * tf.expand_dims(target_mask, axis=1), axis=2)

        mean, var = tf.nn.moments(target_logit, axes=1, keepdims=True)
        std = tf.sqrt(var)
        normalized_logit = (target_logit - mean) / std

        normalized_f = True

        if normalized_f:
            e_grad = tf.reduce_mean(tf.reshape(normalized_logit, [-1, self.n_samples] + [
                                    1] * len(input_shape[1:])) * random_noise, axis=1) / self.search_sigma
        else:
            e_grad = tf.reduce_mean(tf.reshape(target_logit, [-1, self.n_samples] + [
                                    1] * len(input_shape[1:])) * random_noise, axis=1) / self.search_sigma

        self.grad = e_grad
        self.loss = "NES does not need loss to backpropagate"

    def compute_grad(self, x_adv, y_victim, y_target, sess, x):
        fd = None
        assert(self.Y_target_in is not None)
        if self.Y_target_in is None:
            fd = {self.X_in: x_adv, self.Y_in: y_victim}
        else:
            fd = {self.X_in: x_adv, self.Y_in: y_victim,
                  self.Y_target_in: y_target}
        grad = batch_wise_sess_run(sess, self.grad, fd, None, 20)
        return
