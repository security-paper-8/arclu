from datetime import datetime

import os
import numpy as np

import torch
import torch.utils.data
from common import compute_thresholds_each_class, get_y_mu_encoding, split_x_adv, compute_class_mean, test_model, load_data
from models import OursDataParallel, LightImageNetModel
import attacks
from trainers.trainer import MNISTTrainer

import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter


class LightImageNetTrainer(MNISTTrainer):
    def __init__(self, args):
        self.args = args

        self.eps_type = "linf"
        self.min_val = -2.086012125015259
        self.max_val = 2.663571834564209

        self.eps = 16 / 255.0 * (self.max_val - self.min_val)
        self.step_size = 1 / 255.0 * (self.max_val - self.min_val)
        self.iters = 30

        self.l2_test_eps = 1.0
        self.l2_test_step_size = 0.1
        self.l2_test_iters = 40

        self.linf_test_eps = 16 / 255.0
        self.linf_test_step_size = 2 / 255.0
        self.linf_test_iters = 40

        self.num_latents = 1000
        self.num_classes = 20
        self.y_mu_encoding = get_y_mu_encoding(
            self.num_classes, self.num_latents, M=10)

        self.model = OursDataParallel(LightImageNetModel(self.y_mu_encoding, num_classes=self.num_classes),
                                      device_ids=range(torch.cuda.device_count()))

        self.start_k = 0
        self.k = 1

    def epoch_start_handler(self, e):
        pass

    def batch_start_handler(self, e, i):
        pass

    def batch_end_handler(self, e, i, x_adv, y_data):
        pass

    def epoch_end_handler(self, e):
        pass

    def write_robustness_info(self, e, i, save_dir, test_acc, linf_success_ratio, l2_success_ratio):
        # Save model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        os.system("echo \"{}-{}, {:.02f}, {:.02f}, {:.02f}\" >> {}".format(e, i, test_acc,
                                                                           linf_success_ratio, l2_success_ratio, os.path.join(save_dir, "robustness_info.txt")))

    def save_model(self, e, i, model, save_dir, ASR, args):
        # Save model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(
            save_dir, "{}_{}_{}_{}.pth".format(args.dataset, e, i, ASR)))

    def check_best(self, model, train_loader, test_loader, attack,
                   thresholds_each_class, best_geo_mean):
        model.eval()
        test_acc, _, _ = test_model(model, test_loader)

        NUM_TEST_BATCH = 4
        total = 0
        l2_success_num = 0
        linf_success_num = 0

        for i, (x_data, y_data) in enumerate(test_loader):
            if i < NUM_TEST_BATCH:
                x_data = x_data.cuda()
                y_data = y_data.cuda()
                x_lnif_adv_data = attack.perturb(x_data, y_data, None, "linf", self.linf_test_eps,
                                                 self.linf_test_step_size, self.linf_test_iters, self.min_val, self.max_val)
                x_l2_adv_data = attack.perturb(x_data, y_data, None, "l2", self.l2_test_eps,
                                               self.l2_test_step_size, self.l2_test_iters, self.min_val, self.max_val)
                successful_x_linf_adv, _, _, _ = \
                    split_x_adv(model, x_lnif_adv_data,
                                y_data, thresholds_each_class)
                successful_x_l2_adv, _, _, _ = \
                    split_x_adv(model, x_l2_adv_data, y_data,
                                thresholds_each_class)
                l2_success_num += successful_x_l2_adv.size(0)
                linf_success_num += successful_x_linf_adv.size(0)
                total += x_data.size(0)
            else:
                break
        l2_success_ratio = l2_success_num / total
        linf_success_ratio = linf_success_num / total

        geo_mean = (test_acc * (1 - l2_success_ratio) *
                    (1 - linf_success_ratio))**(1.0 / 3.0)
        if geo_mean > best_geo_mean:
            best_geo_mean = geo_mean

        return best_geo_mean, test_acc, linf_success_ratio, l2_success_ratio

    def get_top_k_best_adv(self, x_data, y_data, model, attack, thresholds_each_class, num_classes, start_k, k):
        y_dist_list, y_target_list = torch.topk(model.compute_squared_distance(model(x_data)) + torch.eye(num_classes)[y_data].cuda() * 999999999,
                                                k=k, dim=1, largest=False, sorted=True)
        x_adv_data = x_data.clone()

        not_fooled = torch.ones(x_adv_data.size(0), dtype=torch.bool).cuda()
        score = torch.ones(x_adv_data.size(
            0), dtype=torch.float).cuda() * 999999

        thresholds_each_class = torch.tensor(
            thresholds_each_class).float().cuda()
        for i in range(start_k, k):
            y_target = y_target_list[:, i]

            x_adv_i = attack.perturb(x_data, y_data, y_target, self.eps_type,
                                     self.eps, self.step_size, self.iters, self.min_val, self.max_val)
            with torch.no_grad():
                dist_i = model.compute_squared_distance(model(x_adv_i))
                pred_i = dist_i.argmin(dim=1)
                new_score = dist_i[torch.arange(dist_i.size(
                    0)), pred_i] / thresholds_each_class[pred_i]

                u_indice = not_fooled | (new_score < score)
                x_adv_data[u_indice] = x_adv_i[u_indice]

                score[u_indice] = new_score[u_indice]
                not_fooled[pred_i != y_data] = False

            if not_fooled.float().sum() == 0:
                print(i, "times", (pred_i != y_data).float().sum())
                break
        return x_adv_data

    def train(self):
        writer = SummaryWriter()
        INITIAL_PATH = "checkpoints/light_imagenet_initial_trained.pth"

        args = self.args

        now = datetime.now()
        time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        save_dir = os.path.join(
            "checkpoints", "{}_{}".format(args.dataset, time_str))

        self.time_str = time_str
        self.save_dir = save_dir

        model = self.model
        y_mu_encoding = self.y_mu_encoding

        train_loader, test_loader, thresholds_update_test_loader = load_data(
            args)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()

        model.cuda()
        attack = attacks.PGDOursAttack(model)
        thresholds_each_class = np.array([999999] * self.num_classes)

        # Initial training
        if os.path.exists(INITIAL_PATH):
            model.load_state_dict(torch.load(INITIAL_PATH))
            print("Initial model loaded")
        else:
            print("Initial training")
            for e in range(10):
                model.train()
                print("epoch", e)
                for i, (x_data, y_data) in enumerate(train_loader):
                    x_data = x_data.cuda()
                    y_data = y_data.cuda()

                    optimizer.zero_grad()
                    bo = model(x_data)
                    loss_benign = criterion(bo, y_mu_encoding[y_data])
                    loss_benign.backward()
                    optimizer.step()

                model.eval()
                test_model(model, test_loader)
            torch.save(model.state_dict(), INITIAL_PATH)

        # Adv training
        print("Start adv training")
        schedule_epochs = [0, 100, 200, 300]
        schedule_iter = [30, 60, 90]
        best_geo_mean = 0

        ASR = -1
        for e in range(args.epochs):
            self.epoch_start_handler(e)
            for i, (x_data, y_data) in enumerate(train_loader):
                self.batch_start_handler(e, i)
                model.eval()
                if (i) % args.THRESHOLD_UPDATE_PERIOD_IN_STEP == 0:
                    with torch.no_grad():
                        # Update threshodls_each_class
                        thresholds_each_class = compute_thresholds_each_class(model, thresholds_update_test_loader, FPR=args.FPR,
                                                                              num_batch=args.NUM_THRESHOLDS_UPDATE_BATCH, old_thresholds_each_class=thresholds_each_class, num_classes=self.num_classes)
                        print(np.sqrt(thresholds_each_class))

                if i % 20 == 0:
                    new_geo_mean, test_acc, linf_success_ratio, l2_success_ratio =\
                        self.check_best(model, train_loader, test_loader, attack,
                                        thresholds_each_class, best_geo_mean)
                    writer.add_scalar(
                        'metrics/geo_mean', new_geo_mean, e * len(train_loader) + i * x_data.size(0))
                    if new_geo_mean > best_geo_mean:
                        best_geo_mean = new_geo_mean
                        self.write_robustness_info(
                            e, i, save_dir, test_acc, linf_success_ratio, l2_success_ratio)
                        self.save_model(e, i, model, save_dir, ASR, args)
                        print("Save best, test_acc: {:.02f}, linf: {:.02f}, l2: {:.02f}".format(
                            test_acc, linf_success_ratio, l2_success_ratio))

                # Update attack parameters
                for j in range(len(schedule_iter)):
                    if e == schedule_epochs[j]:
                        self.iters = schedule_iter[j]
                        print("iteration scheduled to {}".format(self.iters))
                # Start
                x_data = x_data.cuda()
                y_data = y_data.cuda()

                # Generate adversarial examples

                x_adv_data = attack.perturb(x_data, y_data, None, self.eps_type, self.eps, self.step_size,
                                            self.iters, self.min_val, self.max_val, random_perturb_start=False)

                model.y_mu_encoding = y_mu_encoding
                successful_x_adv, failed_x_adv, failed_y_true, success_whr = \
                    split_x_adv(model, x_adv_data, y_data,
                                thresholds_each_class)

                ASR = successful_x_adv.size(0) / x_data.size(0)
                print("ASR:", ASR)

                model.y_mu_encoding = y_mu_encoding

                self.batch_end_handler(
                    e, i, successful_x_adv_small, y_data[success_whr_small])
                writer.add_scalar('metrics/ASR', ASR, e *
                                  len(train_loader) + i * x_data.size(0))
                if e % 20 == 0 and i == 0:
                    self.save_model(e, 999, model, save_dir, ASR, args)
                if successful_x_adv.size(0) == 0:
                    print("ASR is zero and continue")
                    continue

                x_benign_all = torch.cat([x_data, failed_x_adv], dim=0)
                y_benign_all = torch.cat([y_data, failed_y_true], dim=0)

                model.train()
                optimizer.zero_grad()
                loss_benign = criterion(
                    model(x_benign_all), y_mu_encoding[y_benign_all])

                if successful_x_adv.size(0) != 0:
                    loss_adv = criterion(model(successful_x_adv),
                                         0.0 * torch.ones((successful_x_adv.size(0), self.num_latents)).cuda())
                    weight = successful_x_adv.size(
                        0) / (successful_x_adv.size(0) + x_benign_all.size(0)) * 0.001
                    loss = loss_benign + loss_adv * weight
                else:
                    loss = loss_benign

                loss.backward()
                optimizer.step()
            self.epoch_end_handler(e)
