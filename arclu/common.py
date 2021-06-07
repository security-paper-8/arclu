import os
import math
import numpy as np
import torch
import torchvision.utils
from torchvision.transforms import transforms


def get_y_mu_encoding(num_classes=10, latent_dim=10, M=10.0):

    y_mu_encoding = np.zeros((num_classes, latent_dim), dtype=np.float32)
    class_latent_dim = latent_dim // num_classes
    for c in range(num_classes):
        y_mu_encoding[c, c * class_latent_dim:(c + 1) * class_latent_dim] = M
    y_mu_encoding = torch.from_numpy(y_mu_encoding).cuda()
    return y_mu_encoding


def compute_class_mean(model, test_loader, num_classes=10):
    reps = [[] for i in range(num_classes)]
    model.eval()
    model.cuda()
    total = 0
    with torch.no_grad():
        for i, (x_data, y_data) in enumerate(test_loader):
            x_data = x_data.cuda()
            y_data = y_data.cuda()
            rep_batch = model(x_data)
            for j in range(x_data.size(0)):
                reps[y_data[j].cpu().item()].append(rep_batch[j].cpu().numpy())
            total += x_data.size(0)

    total = 0
    for i in range(num_classes):
        total += np.array(reps[i]).shape[0]

    return torch.from_numpy(np.array([np.mean(d, axis=0) for d in reps]))


def compute_thresholds_each_class(model, test_loader, FPR, num_batch=10, old_thresholds_each_class=None, num_classes=10):
    model.eval()
    dist_each_class = [[] for _ in range(num_classes)]
    for i, (x_data, y_data) in enumerate(test_loader):
        x_data = x_data.cuda()
        y_data = y_data.cuda()
        z_data = []
        for j in range(math.ceil(x_data.size(0) / 32)):
            z_data.append(model(x_data[j * 32:(j + 1) * 32]).detach())
        z_data = torch.cat(z_data, dim=0)
        dist = model.compute_squared_distance(z_data)
        pred = dist.min(dim=1)[1]

        correct_dist = dist[pred == y_data]
        correct_y_data = y_data[pred == y_data]
        correct_dist_c = correct_dist[torch.arange(
            correct_dist.size(0)), correct_y_data]
        for j in range(correct_dist_c.size(0)):
            dist_each_class[correct_y_data[j].cpu().item()].append(
                correct_dist_c[j].cpu().item())
        if i > num_batch:
            break
    thresholds_each_class = []
    for i, d in enumerate(dist_each_class):
        if len(d) != 0:
            thresholds_each_class.append(np.percentile(d, 100 - FPR))
        else:
            thresholds_each_class.append(old_thresholds_each_class[i])
    return np.array(thresholds_each_class)


def split_x_adv(model, x_adv, y_true, thresholds_each_class):
    dist = []
    for j in range(math.ceil(x_adv.size(0) / 32)):
        j_start = j * 32
        j_end = min((j + 1) * 32, x_adv.size(0))
        dist.append(model.compute_squared_distance(
            model(x_adv[j * 32:(j + 1) * 32])).detach())
    dist = torch.cat(dist, dim=0)

    pred = dist.min(dim=1)[1]
    thresholds_each_class = torch.from_numpy(thresholds_each_class).cuda()
    mislead_success = pred != y_true
    bypass_success = dist[torch.arange(
        dist.size(0)), pred] < thresholds_each_class[pred]

    success_whr = (mislead_success & bypass_success).detach()
    successful_x_adv = x_adv.cpu()[success_whr.cpu()]

    failed_whr = ~(mislead_success & bypass_success)
    failed_x_adv = x_adv[failed_whr]
    failed_y_true = y_true[failed_whr]
    print("mislead success ratio:", mislead_success.float().mean())
    return successful_x_adv, failed_x_adv, failed_y_true, success_whr


def test_model(model, test_loader):
    total = 0
    num_correct = 0
    model.eval()
    model.cuda()
    min_val = 9999
    max_val = -9999
    with torch.no_grad():
        for i, (x_data, y_data) in enumerate(test_loader):
            x_data = x_data.cuda()
            y_data = y_data.cuda()
            pred = model.compute_squared_distance(model(x_data)).min(dim=1)[1]
            num_correct += (pred == y_data).float().sum()
            total += x_data.size(0)
            min_batch, max_batch = x_data.min().item(), x_data.max().item()
            if min_val > min_batch:
                min_val = min_batch
            if max_val < max_batch:
                max_val = max_batch
    acc = num_correct / total
    print("Test accuracy: {}".format(num_correct / total))
    return acc, min_val, max_val


def load_data(args):
    if args.dataset == "mnist":
        trans = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.MNIST(
            "./data", train=True, transform=trans, download=True)
        test_dataset = torchvision.datasets.MNIST(
            "./data", train=False, transform=trans, download=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False)
        thresholds_update_test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True)
        return train_loader, test_loader, thresholds_update_test_loader
    elif args.dataset == "light_imagenet":
        data_dir = '../pytorch-tiny-imagenet/tiny-224/'
        num_workers = {'train': 100, 'val': 0, 'test': 0}
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [
                                     0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [
                                     0.2302, 0.2265, 0.2262]),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [
                                     0.2302, 0.2265, 0.2262]),
            ])
        }
        image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'test', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False, num_workers=num_workers[x])
                       for x in ['train', 'test', 'val']}
        thresholds_update_test_loader = torch.utils.data.DataLoader(image_datasets['test'],
                                                                    batch_size=args.batch_size, shuffle=False, num_workers=num_workers['test'])

        dataset_sizes = {x: len(image_datasets[x])
                         for x in ['train', 'test', 'val']}

        return dataloaders['train'], dataloaders['test'], thresholds_update_test_loader


# For kNN atack
def get_undetected_representations(model, test_loader, FPR, num_classes):
    threshold_each_class = torch.tensor(compute_thresholds_each_class(model, test_loader, FPR,
                                                                      num_batch=10, old_thresholds_each_class=None,
                                                                      num_classes=num_classes)).cuda()
    undetected_representation_list = []
    undetected_target_list = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.cuda()
            target = target.cuda()
            representation = model(data)
            dist = model.compute_squared_distance(representation)
            dist_c = dist.gather(1, target.view(-1, 1)).squeeze(1)

            undetected_benign_whr = dist_c < threshold_each_class[target]
            undetected_representation = representation[undetected_benign_whr]
            undetected_target = target[undetected_benign_whr]

            undetected_representation_list.append(undetected_representation)
            undetected_target_list.append(undetected_target)

    undetected_representation_all = torch.cat(
        undetected_representation_list, dim=0)
    undetected_target_all = torch.cat(undetected_target_list, dim=0)

    return undetected_representation_all, undetected_target_all


def get_k_nearest_representations(model, test_loader, k, num_data, undetected_representation, undetected_target, is_targeted, num_classes):
    total = 0
    data_list = []
    target_list = []
    k_nearest_representation_list = []
    undetected_representation = undetected_representation[:1000]
    undetected_target = undetected_target[:1000]
    with torch.no_grad():
        # For each batch
        for i, (data, target) in enumerate(test_loader):
            print("batch", i)
            data = data.cuda()
            target = target.cuda()
            adv_target = (target + torch.randint(1, num_classes,
                                                 size=target.size()).cuda()) % num_classes
            representation = model(data)
            k_nearest_representation = [[]
                                        for i in range(representation.size(0))]
            k_nearest_distance = [[] for i in range(representation.size(0))]
            # Measure distance to undetected (valid) representation
            for j in range(len(undetected_representation)):
                test_representation = undetected_representation[j]
                distance = torch.pow(
                    representation - test_representation.unsqueeze(0), 2).sum(dim=1)
                for l in range(representation.size(0)):
                    if is_targeted is False:
                        if target[l] == undetected_target[j]:
                            # Label of nearest neighbor should be different from that of data
                            continue
                    elif is_targeted is True:
                        if adv_target[l] != undetected_target[j]:
                            # Label of nearest neighbor should be different from that of data
                            continue
                    if len(k_nearest_distance[l]) == 0:
                        # No nearest neighbor found, just add current ones for initialization
                        k_nearest_distance[l].append(distance[l])
                        k_nearest_representation[l].append(
                            test_representation.cpu().numpy())
                    else:
                        inserted = False
                        for m in range(len(k_nearest_distance[l])):
                            if k_nearest_distance[l][m] > distance[l]:
                                k_nearest_distance[l].insert(m, distance[l])
                                k_nearest_representation[l].insert(
                                    m, test_representation.cpu().numpy())
                                inserted = True
                                break
                        if inserted is False and len(k_nearest_distance[l]) < k:
                            k_nearest_distance[l].append(distance[l])
                            k_nearest_representation[l].append(
                                test_representation.cpu().numpy())

                        # Trim for length of nearest list to be k
                        k_nearest_distance[l] = k_nearest_distance[l][:k]
                        k_nearest_representation[l] = k_nearest_representation[l][:k]

            k_nearest_representation_list.append(
                torch.tensor(k_nearest_representation))
            data_list.append(data)
            target_list.append(target)

            total += data.size(0)
            if total > num_data:
                break

    data_all = torch.cat(data_list, dim=0)
    target_all = torch.cat(target_list, dim=0)
    k_nearest_representation_all = torch.cat(
        k_nearest_representation_list, dim=0)

    dataset = torch.utils.data.TensorDataset(
        data_all, target_all, k_nearest_representation_all)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    return loader
