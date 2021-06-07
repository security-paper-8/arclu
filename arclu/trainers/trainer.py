from datetime import datetime

import os
import numpy as np

import torch
import torch.utils.data
from common import compute_thresholds_each_class, get_y_mu_encoding, split_x_adv, compute_class_mean, test_model, load_data
from models import MNISTModel, OursDataParallel
import attacks
from torch.utils.tensorboard import SummaryWriter


class MNISTTrainer:
    def __init__(self, args):
        self.args = args

        self.eps_type = "linf"
        self.eps = 0.3
        self.step_size = 0.01
        self.iters = 40

        self.min_val = 0.0
        self.max_val = 1.0

        self.l2_test_eps = 3.0
        self.l2_test_step_size = 0.1
        self.l2_test_iters = 400

        self.linf_test_eps = 0.3
        self.linf_test_step_size = 0.001
        self.linf_test_iters = 400

        self.y_mu_encoding = get_y_mu_encoding(M=10)
        self.model = MNISTModel(self.y_mu_encoding)

    def epoch_start_handler(self, e):
        pass

    def batch_start_handler(self, e, i):
        pass

    def batch_end_handler(self, e, i, x_adv, y_data):
        pass

    def epoch_end_handler(self, e):
        pass

    def train(self):
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
        thresholds_each_class = np.array([999999] * 10)

        best_geo_mean = 0
        for e in range(args.epochs):
            self.epoch_start_handler(e)
            if e == 30:
                print("eps update!!")
                self.step_size = 0.001
                self.iters = 400

            for i, (x_data, y_data) in enumerate(train_loader):
                self.batch_start_handler(e, i)
                x_data = x_data.cuda()
                y_data = y_data.cuda()

                model.eval()

                x_adv_data = attack.perturb(x_data, y_data, None, self.eps_type,
                                            self.eps, self.step_size, self.iters, self.min_val, self.max_val)

                successful_x_adv, failed_x_adv, failed_y_true, success_whr = \
                    split_x_adv(model, x_adv_data, y_data,
                                thresholds_each_class)

                print("ASR:", successful_x_adv.size(0) / x_data.size(0))
                self.batch_end_handler(
                    e, i, successful_x_adv, y_data[success_whr])
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
                    loss_adv = criterion(model(successful_x_adv.cuda()),
                                         0.0 * torch.ones((successful_x_adv.size(0), self.model.y_mu_encoding.shape[1])).cuda())

                    loss = loss_benign + loss_adv
                else:
                    loss = loss_benign
                loss.backward()
                optimizer.step()

                model.eval()
                if (i + 1) % args.THRESHOLD_UPDATE_PERIOD_IN_STEP == 0:
                    # Update threshodls_each_class
                    thresholds_each_class = compute_thresholds_each_class(model, thresholds_update_test_loader,
                                                                          FPR=args.FPR, num_batch=args.NUM_THRESHOLDS_UPDATE_BATCH, old_thresholds_each_class=thresholds_each_class)
                    print(np.sqrt(thresholds_each_class))

            model.eval()
            test_acc, _, _ = test_model(model, test_loader)

            # Save model
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(
                save_dir, "{}_{}.pth".format(args.dataset, e)))

            NUM_TEST_BATCH = 10
            total = 0
            l2_success_num = 0
            linf_success_num = 0

            thresholds_each_class = compute_thresholds_each_class(model, thresholds_update_test_loader,
                                                                  FPR=1.0,
                                                                  num_batch=args.NUM_THRESHOLDS_UPDATE_BATCH, old_thresholds_each_class=thresholds_each_class)
            print(np.sqrt(thresholds_each_class))

            for i, (x_data, y_data) in enumerate(train_loader):
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
            print("l2_success_ratio", l2_success_ratio,
                  "linf_success_ratio", linf_success_ratio)
            geo_mean = (test_acc * (1 - l2_success_ratio) *
                        (1 - linf_success_ratio))**(1.0 / 3.0)
            if geo_mean > best_geo_mean:
                torch.save(model.state_dict(), os.path.join(
                    save_dir, "{}_best_{}.pth".format(self.args.dataset, e)))
                best_geo_mean = geo_mean

            self.epoch_end_handler(e)


class MNISTTrainerForPlotingClusters(MNISTTrainer):
    def epoch_start_handler(self, e):
        self.x_adv_list = []
        self.y_true_list = []

    def batch_start_handler(self, e, i):
        pass

    def batch_end_handler(self, e, i, x_adv, y_data):
        self.x_adv_list.append(x_adv)
        self.y_true_list.append(y_data)

    def epoch_end_handler(self, e):
        x_adv = torch.cat(self.x_adv_list, dim=0).cpu().numpy()
        y_true = torch.cat(self.y_true_list, dim=0).cpu().numpy()
        np.savez(os.path.join(self.save_dir, "adv_data_{}.npz".format(e)),
                 x_adv=x_adv, y_true=y_true)
