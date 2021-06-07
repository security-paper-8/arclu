import sys
sys.path.append("../ar_comparison/eval_codes/attacks/")
import math
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from autoattack import AutoAttack
from models import MNISTModel, LightImageNetModel, OursDataParallel
from attacks import PGDOursAttack, NESOursAttack, TransferOursAttack, BoundaryOursAttack, ModelForFoolbox,\
    ModelForRescale, PGDDisperseOursAttack,\
    RepresentationOursAttack
from common import get_y_mu_encoding, split_x_adv, compute_thresholds_each_class, compute_class_mean,\
    test_model, load_data, get_undetected_representations,\
    get_k_nearest_representations
import attack_common


def pgd_attack(model, test_loader, min_value, max_value, args):
    attack = PGDOursAttack(model)

    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps
    step_size = args.step_size * scale_factor
    max_iters = args.iters
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    random_epsilon_dict = {'mnist': 0.0001, 'light_imagenet': 0.1}
    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        if targeted:
            y_victim_target = (
                y_victim + torch.randint(1, args.NUM_CLASSES, size=y_victim.size()).cuda()) % args.NUM_CLASSES
            x_adv_batch = attack.perturb(x_victim, y_victim, y_victim_target, eps_type,
                                         epsilon, step_size, max_iters, min_value, max_value, device="cuda:0", repetition=args.repetition,
                                         multi_targeted=args.multi_targeted, random_perturb_start=args.random_start, random_epsilon=random_epsilon_dict[args.dataset])
        else:
            x_adv_batch = attack.perturb(x_victim, y_victim, None, eps_type, epsilon,
                                         step_size, max_iters, min_value, max_value, device="cuda:0", repetition=args.repetition,
                                         multi_targeted=args.multi_targeted, random_perturb_start=args.random_start, random_epsilon=random_epsilon_dict[args.dataset])

        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_benign.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break
    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def pgd_disperse_attack(model, test_loader, min_value, max_value, args):
    print("M:", model.y_mu_encoding[0].max())
    print("n_latent:", model.y_mu_encoding[0].size(0))
    print("n_class:", len(model.y_mu_encoding))
    print("zero_threshold_multiplier:", args.zero_threshold_multiplier)
    print("adv_coeff:", args.adv_coeff)
    attack = PGDDisperseOursAttack(model, M=model.y_mu_encoding[0].max(),
                                   n_latent=model.y_mu_encoding[0].size(0),
                                   n_class=len(model.y_mu_encoding),
                                   zero_threshold_multiplier=args.zero_threshold_multiplier,
                                   adv_coeff=args.adv_coeff)

    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps
    step_size = args.step_size * scale_factor
    max_iters = args.iters
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        if targeted:
            y_victim_target = (
                y_victim + torch.randint(1, args.NUM_CLASSES, size=y_victim.size()).cuda()) % args.NUM_CLASSES
            x_adv_batch = attack.perturb(x_victim, y_victim, y_victim_target, eps_type,
                                         epsilon, step_size, max_iters, min_value, max_value, device="cuda:0", repetition=args.repetition,
                                         multi_targeted=args.multi_targeted, random_perturb_start=args.random_start)
        else:
            x_adv_batch = attack.perturb(x_victim, y_victim, None, eps_type, epsilon,
                                         step_size, max_iters, min_value, max_value, device="cuda:0", repetition=args.repetition,
                                         multi_targeted=args.multi_targeted, random_perturb_start=args.random_start)

        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_benign.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break

    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def nes_attack(model, test_loader, min_value, max_value, args):
    attack = NESOursAttack(
        model, n_samples=args.n_sample, search_sigma=args.search_sigma)

    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps
    step_size = args.step_size * scale_factor
    max_iters = args.iters

    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0
    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        y_victim_target = (y_victim + torch.randint(1, args.NUM_CLASSES,
                                                    size=y_victim.size()).cuda()) % args.NUM_CLASSES
        x_adv_batch = attack.perturb(x_victim, y_victim, y_victim_target, eps_type,
                                     epsilon, step_size, max_iters, min_value, max_value, device="cuda:0")
        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_benign.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break
    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def boundary_attack(model, test_loader, min_value, max_value, args):
    thresholds_each_class = compute_thresholds_each_class(model, test_loader, 5, num_batch=50,
                                                          num_classes=args.NUM_CLASSES)
    attack = BoundaryOursAttack(
        model, n_delta=args.n_delta, thresholds_each_class=thresholds_each_class)

    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps
    init_delta = args.init_delta
    init_epsilon = args.init_epsilon
    max_iters = args.iters
    print("n_delta:", type(args.n_delta))
    print("min_value :", min_value)
    print("max_value :", max_value)

    x_adv = []
    x_ben = []
    y_true = []
    total = 0
    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()

        x_adv_batch = attack.perturb(
            x_victim, y_victim, eps_type, epsilon, init_delta, init_epsilon, max_iters, min_value, max_value)
        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_ben.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break
    return np.concatenate(x_adv, axis=0), np.concatenate(x_ben, axis=0), np.concatenate(y_true, axis=0)


def transfer_attack(model, test_loader, min_value, max_value, args):
    thresholds_each_class = compute_thresholds_each_class(model, test_loader, 5, num_batch=999999,
                                                          num_classes=args.NUM_CLASSES)
    attack = TransferOursAttack(
        args.dataset, model, thresholds_each_class=np.array(thresholds_each_class))

    x_benign, y_true, x_adv = attack.perturb(
        min_value, max_value)
    return x_adv, x_benign, y_true


def cw_attack(model, test_loader, min_value, max_value, args):

    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    assert eps_type == "l2"

    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps

    step_size = args.step_size * scale_factor
    max_iters = args.iters
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    import foolbox
    fmodel = foolbox.models.PyTorchModel(ModelForFoolbox(
        model).eval(), bounds=(min_value, max_value))
    attack = foolbox.attacks.L2CarliniWagnerAttack(binary_search_steps=args.binary_search_steps,
                                                   steps=max_iters, confidence=args.confidence,
                                                   stepsize=step_size, initial_const=args.initial_const, abort_early=False)

    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        _, x_adv_batch, _ = attack(
            fmodel, x_victim, y_victim, epsilons=[epsilon])
        x_adv_batch = x_adv_batch[0]
        x_adv.append(x_adv_batch.cpu().numpy())
        x_benign.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break

    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def mim_attack(model, test_loader, min_value, max_value, args):
    attack = PGDOursAttack(model)

    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps
    step_size = args.step_size * scale_factor
    max_iters = args.iters
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        if targeted:
            y_victim_target = (
                y_victim + torch.randint(1, args.NUM_CLASSES, size=y_victim.size()).cuda()) % args.NUM_CLASSES
            x_adv_batch = attack.perturb(x_victim, y_victim, y_victim_target, eps_type,
                                         epsilon, step_size, max_iters, min_value, max_value, decay=0.9, device="cuda:0")
        else:
            x_adv_batch = attack.perturb(x_victim, y_victim, None, eps_type, epsilon,
                                         step_size, max_iters, min_value, max_value, decay=0.9, device="cuda:0")

        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_benign.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break

    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def ead_attack(model, test_loader, min_value, max_value, args):
    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    assert eps_type == "l1"

    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps

    step_size = args.step_size * scale_factor
    max_iters = args.iters
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    import foolbox
    fmodel = foolbox.models.PyTorchModel(ModelForFoolbox(
        model).eval(), bounds=(min_value, max_value))
    attack = foolbox.attacks.EADAttack(binary_search_steps=args.binary_search_steps,
                                       steps=max_iters,
                                       confidence=args.confidence,
                                       initial_stepsize=step_size,
                                       initial_const=args.initial_const,
                                       decision_rule="EN",
                                       abort_early=False)

    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        _, x_adv_batch, _ = attack(
            fmodel, x_victim, y_victim, epsilons=[epsilon])
        x_adv_batch = x_adv_batch[0]
        x_adv.append(x_adv_batch.cpu().numpy())
        x_benign.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break

    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def deepfool_attack(model, test_loader, min_value, max_value, args):
    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    assert eps_type == "linf"
    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps

    max_iters = args.iters
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    import foolbox
    fmodel = foolbox.models.PyTorchModel(ModelForFoolbox(
        model).eval(), bounds=(min_value, max_value))
    """
        self,
        *,
        steps: int = 50,
        candidates: Optional[int] = 10,
        overshoot: float = 0.02,
        loss: Union[Literal["logits"], Literal["crossentropy"]] = "logits",
    """
    attack = foolbox.attacks.LinfDeepFoolAttack(steps=max_iters,
                                                overshoot=0.2,
                                                loss="logits")

    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        _, x_adv_batch, _ = attack(
            fmodel, x_victim, y_victim, epsilons=[epsilon])
        x_adv_batch = x_adv_batch[0]
        x_adv.append(x_adv_batch.cpu().numpy())
        x_benign.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break

    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def fgsm_attack(model, test_loader, min_value, max_value, args):
    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    assert eps_type == "linf"
    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps
    step_size = args.step_size * scale_factor
    max_iters = args.iters
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    import foolbox
    fmodel = foolbox.models.PyTorchModel(ModelForFoolbox(
        model).eval(), bounds=(min_value, max_value))
    """
        self,
        *,
        steps: int = 50,
        candidates: Optional[int] = 10,
        overshoot: float = 0.02,
        loss: Union[Literal["logits"], Literal["crossentropy"]] = "logits",
    """
    attack = foolbox.attacks.FGSM()

    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        _, x_adv_batch, _ = attack(
            fmodel, x_victim, y_victim, epsilons=[epsilon])
        x_adv_batch = x_adv_batch[0]
        x_adv.append(x_adv_batch.cpu().numpy())
        x_benign.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break
    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def semantic_test_attack(model, min_val, max_val, args):
    attack = PGDOursAttack(model)
    num_classes = args.NUM_CLASSES
    scale_factor = (max_val - min_val)
    eps_type = args.eps_type
    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps
    step_size = args.step_size * scale_factor
    max_iters = args.iters
    targeted = args.targeted

    print("min_value :", min_val)
    print("max_value :", max_val)
    print("scale factor:", scale_factor)

    def att_func(x_data_numpy, y_data_numpy):
        x_data = torch.from_numpy(x_data_numpy).cuda()
        y_data = torch.from_numpy(y_data_numpy).cuda()

        if targeted:
            y_target = (y_data + torch.randint(1, num_classes,
                                               y_data.size()).cuda()) % num_classes
            x_adv = attack.perturb(
                x_data, y_data, y_target, eps_type, epsilon, step_size, max_iters, min_val, max_val)
        else:
            x_adv = attack.perturb(
                x_data, y_data, None, eps_type, epsilon, step_size, max_iters, min_val, max_val)
        y_adv_pred = model(x_adv).max(dim=1)[1]

        return x_adv.cpu().detach().numpy(), y_adv_pred.cpu().detach().numpy()

    st_attack = attack_common.SemanticTestAttack(
        args.dataset, att_func, batch_size=128, attack_size=args.attack_size)
    xb, yb, xa, ya, xbs, xas = st_attack.perturb(min_val, max_val)
    st_attack.save_result(args.result_path + "result_{}_{}_same_scale.npz".format(
        args.dataset, args.attack), xbs, yb, xas, ya)

    return xa, xb, yb


def knn_attack(model, test_loader, train_loader, min_value, max_value, args):
    attack = RepresentationOursAttack(model)

    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps
    step_size = args.step_size * scale_factor
    max_iters = args.iters
    targeted = args.targeted
    assert(targeted == True)
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    # cross-class nearest neighbor search
    undetected_representation, undetected_target = get_undetected_representations(model, train_loader,
                                                                                  FPR=args.FPRP, num_classes=args.NUM_CLASSES)
    # `is_targeted=True` is more effective
    dataset_with_target_representation = get_k_nearest_representations(model, test_loader, k=args.n_neighbor, num_data=1000,  # 1000,\
                                                                       undetected_representation=undetected_representation, undetected_target=undetected_target,\
                                                                       is_targeted=True, num_classes=args.NUM_CLASSES)

    for _, (data, target, target_representation) in enumerate(dataset_with_target_representation):
        x_victim = data.cuda()
        y_victim = target.cuda()

        # y_target -> target_representation
        x_adv_batch = attack.perturb(x_victim, y_victim, target_representation, eps_type,
                                     epsilon, step_size, max_iters, min_value, max_value, device="cuda:0", repetition=args.repetition,
                                     multi_targeted=args.multi_targeted)

        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_benign.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break
    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def representation_attack(model, test_loader, min_value, max_value, args):
    attack = RepresentationOursAttack(model)

    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    if args.eps_type != "l0":
        epsilon = args.eps * scale_factor
    else:
        epsilon = args.eps
    step_size = args.step_size * scale_factor
    max_iters = args.iters
    targeted = args.targeted
    assert(targeted == True)
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    # cross-class nearest neighbor search
    undetected_representation, undetected_target = get_undetected_representations(model, test_loader,
                                                                                  FPR=args.FPRP, num_classes=args.NUM_CLASSES)
    dataset_with_target_representation = get_k_nearest_representations(model, test_loader, k=10, num_data=1000,
                                                                       undetected_representation=undetected_representation, undetected_target=undetected_target,
                                                                       is_targeted=True, num_classes=args.NUM_CLASSES)

    for _, (data, target, target_representation) in enumerate(dataset_with_target_representation):
        x_victim = data.cuda()
        y_victim = target.cuda()

        # y_target -> target_representation
        x_adv_batch = attack.perturb(x_victim, y_victim, target_representation, eps_type,
                                     epsilon, step_size, max_iters, min_value, max_value, device="cuda:0", repetition=args.repetition,
                                     multi_targeted=args.multi_targeted)

        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_benign.append(x_victim.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += x_victim.size(0)
        if total > args.attack_size:
            break
    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def auto_attack(model, test_loader, min_value, max_value, args):
    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    # We don't need scaling in the auto-attack because we are already in a range 0-1 that autoattack requires.
    epsilon = args.eps
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    fmodel = ModelForRescale(model, min_value, max_value).eval()
    print("epsilon:", epsilon)
    assert(eps_type in ['linf', 'l2'])
    eps_dict = {'linf': 'Linf', 'l2': 'L2'}
    thresholds_each_class = compute_thresholds_each_class(model, test_loader, 5, num_batch=50,
                                                          num_classes=args.NUM_CLASSES)

    attack = AutoAttack(fmodel, norm=eps_dict[eps_type], eps=epsilon,
                        n_target_classes=args.NUM_CLASSES - 1, threshold_each_class=-1 * thresholds_each_class)

    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        # auto attack requires for inputs to scale 0 to 1
        x_victim = (x_victim - min_value) / (max_value - min_value)
        if targeted:
            assert(False)
        else:
            attack.square.seed = attack.get_seed()
            x_adv_batch = attack.run_standard_evaluation(x_victim, y_victim)

        # scale back to the original range
        x_adv_batch = x_adv_batch * (max_value - min_value) + min_value
        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_benign.append(data.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += data.size(0)
        if total > args.attack_size:
            break
    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def square_attack(model, test_loader, min_value, max_value, args):
    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    # We don't need scaling in the auto-attack because we are already in a range 0-1 that autoattack requires.
    epsilon = args.eps
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    fmodel = ModelForRescale(model, min_value, max_value).eval()
    print("epsilon:", epsilon)
    assert(eps_type in ['linf', 'l2'])
    eps_dict = {'linf': 'Linf', 'l2': 'L2'}
    attack = AutoAttack(
        fmodel, norm=eps_dict[eps_type], eps=epsilon, n_target_classes=args.NUM_CLASSES - 1)

    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        # auto attack requires for inputs to scale 0 to 1
        x_victim = (x_victim - min_value) / (max_value - min_value)
        if targeted:
            assert(False)
        else:
            attack.square.seed = attack.get_seed()
            x_adv_batch = attack.square.perturb(x_victim, y_victim)

        # scale back to the original range
        x_adv_batch = x_adv_batch * (max_value - min_value) + min_value
        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_benign.append(data.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += data.size(0)
        if total > args.attack_size:
            break

    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def fabt_attack(model, test_loader, min_value, max_value, args):
    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    # We don't need scaling in the auto-attack because we are already in a range 0-1 that autoattack requires.
    epsilon = args.eps
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    fmodel = ModelForRescale(model, min_value, max_value).eval()
    print("epsilon:", epsilon)
    assert(eps_type in ['linf', 'l2'])
    eps_dict = {'linf': 'Linf', 'l2': 'L2'}
    attack = AutoAttack(
        fmodel, norm=eps_dict[eps_type], eps=epsilon, n_target_classes=args.NUM_CLASSES - 1)

    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        # auto attack requires for inputs to scale 0 to 1
        x_victim = (x_victim - min_value) / (max_value - min_value)
        if targeted:
            assert(False)
        else:
            attack.fab.targeted = True
            attack.fab.seed = attack.get_seed()
            x_adv_batch = attack.fab.perturb(x_victim, y_victim)

        # scale back to the original range
        x_adv_batch = x_adv_batch * (max_value - min_value) + min_value
        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_benign.append(data.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += data.size(0)
        if total > args.attack_size:
            break

    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def apgdt_attack(model, test_loader, min_value, max_value, args):
    scale_factor = (max_value - min_value)
    eps_type = args.eps_type
    # We don't need scaling in the auto-attack because we are already in a range 0-1 that autoattack requires.
    epsilon = args.eps
    targeted = args.targeted
    print("min_value :", min_value)
    print("max_value :", max_value)
    print("scale factor:", scale_factor)
    x_adv = []
    y_true = []
    x_benign = []
    total = 0

    thresholds_each_class = compute_thresholds_each_class(model, test_loader, 5, num_batch=50,
                                                          num_classes=args.NUM_CLASSES)

    fmodel = ModelForRescale(model, min_value, max_value).eval()
    print("epsilon:", epsilon)
    assert(eps_type in ['linf', 'l2'])
    eps_dict = {'linf': 'Linf', 'l2': 'L2'}
    attack = AutoAttack(fmodel, norm=eps_dict[eps_type], eps=epsilon,
                        n_target_classes=args.NUM_CLASSES - 1, threshold_each_class=thresholds_each_class)

    for _, (data, target) in enumerate(test_loader):
        x_victim = data.cuda()
        y_victim = target.cuda()
        # auto attack requires for inputs to scale 0 to 1
        x_victim = (x_victim - min_value) / (max_value - min_value)
        if targeted:
            assert(False)
        else:
            attack.apgd_targeted.seed = attack.get_seed()
            _, x_adv_batch = attack.apgd_targeted.perturb(
                x_victim, y_victim, best_loss=True)

        # scale back to the original range
        x_adv_batch = x_adv_batch * (max_value - min_value) + min_value
        x_adv.append(x_adv_batch.detach().cpu().numpy())
        x_benign.append(data.detach().cpu().numpy())
        y_true.append(y_victim.detach().cpu().numpy())

        total += data.size(0)
        if total > args.attack_size:
            break
    return np.concatenate(x_adv, axis=0), np.concatenate(x_benign, axis=0),\
        np.concatenate(y_true, axis=0)


def save_result(model, test_loader, x_benign,
                y_true, x_adv, args):
    """
    FPR_labels
    FPR_probs
    original_probs
    adv_probs
    true_labels
    adv_images
    """
    batch_size = 128
    if "test_labels" in dir(test_loader.dataset):
        FPR_labels = test_loader.dataset.test_labels
    elif "targets" in dir(test_loader.dataset):
        FPR_labels = test_loader.dataset.targets
    else:
        raise NotImplementedError()
    true_labels = y_true
    adv_images = x_adv
    FPR_probs = []
    model.eval()
    with torch.no_grad():
        for _, (x_data, _) in enumerate(test_loader):
            x_data = x_data.cuda()
            out = model(x_data)
            dist_list = model.compute_squared_distance(out)
            min_dist, pred = dist_list.min(dim=1)
            pred_one_hot = torch.eye(dist_list.size(1))[pred].cuda()
            other_one_hot = 1.0 - pred_one_hot
            FPR_probs.append(-min_dist.unsqueeze(1) *
                             pred_one_hot - other_one_hot * 9999999.0)

            assert (FPR_probs[-1].max(dim=1)[1] == pred).all()

        original_probs = []
        for i in range(int(math.ceil(len(x_benign) / batch_size))):
            max_batch_index = min((i + 1) * batch_size, len(x_benign))
            x_data = torch.from_numpy(
                x_benign[i * batch_size:max_batch_index]).cuda()
            out = model(x_data)

            dist_list = model.compute_squared_distance(out)
            min_dist, pred = dist_list.min(dim=1)

            pred_one_hot = torch.eye(dist_list.size(1))[pred].cuda()
            other_one_hot = 1.0 - pred_one_hot
            original_probs.append(-min_dist.unsqueeze(1)
                                  * pred_one_hot - other_one_hot * 9999999.0)

        adv_probs = []
        for i in range(int(math.ceil(len(x_adv) / batch_size))):
            max_batch_index = min((i + 1) * batch_size, len(x_adv))
            x_data = torch.from_numpy(
                x_adv[i * batch_size:max_batch_index]).cuda()
            out = model(x_data)

            dist_list = model.compute_squared_distance(out)
            min_dist, pred = dist_list.min(dim=1)

            pred_one_hot = torch.eye(dist_list.size(1))[pred].cuda()
            other_one_hot = 1.0 - pred_one_hot
            adv_probs.append(-min_dist.unsqueeze(1) *
                             pred_one_hot - other_one_hot * 9999999.0)

        FPR_probs = torch.cat(FPR_probs, dim=0).cpu().numpy()
        original_probs = torch.cat(original_probs, dim=0).cpu().numpy()
        adv_probs = torch.cat(adv_probs, dim=0).cpu().numpy()

        np.savez(args.result_path + "result_" + args.dataset + "_" + args.attack + ".npz",
                 FPR_labels=FPR_labels[:FPR_probs.shape[0]],
                 FPR_probs=FPR_probs, original_probs=original_probs, true_labels=true_labels,
                 adv_images=adv_images, adv_probs=adv_probs, repetition=args.repetition)


def test(args):
    __train_loader, test_loader, _ = load_data(args)
    if args.dataset == "mnist":
        y_mu_encoding = get_y_mu_encoding(M=10)
        model = MNISTModel(y_mu_encoding)
        model.load_state_dict(torch.load(
            "./checkpoints/mnist_2020_10_08_03_46_22/mnist_best_41.pth"))
    elif args.dataset == "light_imagenet":
        y_mu_encoding = get_y_mu_encoding(args.NUM_CLASSES, 1000, 10)
        model = OursDataParallel(LightImageNetModel(y_mu_encoding, num_classes=args.NUM_CLASSES),
                                 device_ids=range(torch.cuda.device_count()))
        if args.ckpt == None:
            model.load_state_dict(torch.load(
                "./checkpoints/light_imagenet_2021_03_02_16_14_15/light_imagenet_160_999_0.385.pth"))
        else:
            model.load_state_dict(torch.load(args.ckpt))
        model = model.module
    else:
        raise NotImplementedError()

    _, min_val, max_val = test_model(model, test_loader)
    is_successed = None
    if args.attack == "pgd":
        x_adv, x_benign, y_true = pgd_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "nes":
        x_adv, x_benign, y_true = nes_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "boundary":
        x_adv, x_benign, y_true = boundary_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "transfer":
        x_adv, x_benign, y_true = transfer_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "semantic":
        x_adv, x_benign, y_true = semantic_test_attack(
            model, min_val, max_val, args)
    elif args.attack == "cw":
        x_adv, x_benign, y_true = cw_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "mim":
        x_adv, x_benign, y_true = mim_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "ead":
        x_adv, x_benign, y_true = ead_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "deepfool":
        x_adv, x_benign, y_true = deepfool_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "fgsm":
        x_adv, x_benign, y_true = fgsm_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "knn":
        x_adv, x_benign, y_true = knn_attack(
            model, test_loader, __train_loader, min_val, max_val, args)
    elif args.attack == "representation":
        x_adv, x_benign, y_true = representation_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "square":
        x_adv, x_benign, y_true = square_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "fabt":
        x_adv, x_benign, y_true = fabt_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "apgdt":
        x_adv, x_benign, y_true = apgdt_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "auto":
        x_adv, x_benign, y_true = auto_attack(
            model, test_loader, min_val, max_val, args)
    elif args.attack == "pgddisperse":
        x_adv, x_benign, y_true = pgd_disperse_attack(
            model, test_loader, min_val, max_val, args)
    else:
        raise NotImplementedError()

    if args.dataset != "light_imagenet":
        save_result(model, test_loader, x_benign, y_true, x_adv, args)
    else:
        save_result(model, test_loader, x_benign, y_true, x_adv, args)


def main():
    # mnist_test()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=['mnist', 'light_imagenet'])
    parser.add_argument(
        '--attack', choices=['pgd', 'nes', 'boundary', 'transfer', 'semantic', 'cw', 'mim', 'ead', 'deepfool', 'fgsm',
                             'knn', 'representation', 'square', 'fabt', 'apgdt', 'auto', 'pgddisperse'])

    parser.add_argument("--FPRP", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--repetition", type=int, default=1)

    # PGD, NES
    parser.add_argument("--eps_type", type=str,
                        choices=["l0", "l1", "l2", "linf"])
    parser.add_argument("--eps", type=float)
    parser.add_argument("--step_size", type=float)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--targeted", action="store_true", default=False)
    parser.add_argument("--multi_targeted", action="store_true", default=False)
    parser.add_argument("--random_start", action="store_true", default=False)

    # NES
    parser.add_argument("--n_sample", type=int)
    parser.add_argument("--search_sigma", type=float)

    # Boundary
    parser.add_argument("--n_delta", type=int)
    parser.add_argument("--init_delta", type=float)
    parser.add_argument("--init_epsilon", type=float)

    parser.add_argument('--result_path', type=str, default="./vr_result/")

    parser.add_argument('--attack_size', type=int, default=10000)

    # CW
    parser.add_argument("--initial_const", type=float)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--binary_search_steps", type=int)

    # KNN
    parser.add_argument("--n_neighbor", type=int, default=20)

    # PGDDisperse
    parser.add_argument("--zero_threshold_multiplier", type=float, default=None)
    parser.add_argument("--adv_coeff", type=float, default=None)
    parser.add_argument(
        '--ckpt', type=str, default=None)

    args = parser.parse_args()
    args.NUM_CLASSES = 10 if args.dataset != "light_imagenet" else 20
    import os
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path, exist_ok=True)
    test(args)


if __name__ == "__main__":
    main()
