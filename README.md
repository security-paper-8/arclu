# ARCLU (Adversarial Representations as CLUsters), robust detector against adaptive whitebox attacks.
This is a source code for reproducing a paper __"ARCLU: Learning to Separate Clusters of Adversarial Representationsfor Robust Adversarial Detection"__.

ARCLU finds overlapping adversarial representations and separates them at a new cluster.

The cluster separation leads to robust detection against adaptive whitebox attacks.

# Installation
## 1. Project clone
```
git clone https://github.com/security-paper-8/arclu.git && cd arclu
```
## 2. Environment setting
```
bash ./install_scripts/set_env.sh && conda activate envtest_arclu
```
## 3. Dataset download (Optional for LightImageNet)
```
bash ./install_scripts/get_light_imagenet.sh
```
## 4. Quick test
### 4-1. Attack (![](http://www.sciweavers.org/tex2img.php?eq=%5Cell_%5Cinfty%3D0.3&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0), on MNIST)
```
bash ./install_scripts/quick_test.sh
```
### 4-1. Output
```
arclu,  0.1228, 0.9690
```
It indicates ARCLU achieved ASR 0.1228 and EROC 0.9690.

# Test
There are many test scripts in `ar_comparison/eval_codes/experiments/scripts/`.

To execute the scripts:
```
# Change directory
cd ar_comparison/eval_codes/experiments
# Conduct the mnist quick experiment.
python ./scripts/mnist_quick_test.py
```
You can conduct different experiment with:
```
python ./scripts/<script_name_here>
```
A test script generates result files in `ar_comparison/eval_codes/experiments/results/`.

To compute the final performances from a result file:
```
python ./report.py ./results/<exp_path_here>
```
You can get the `./results/<exp_path_here>` in the corresponding script file by concatenating ROOT_PATH and EXP_PATH.

In case of the `./scripts/mnist_quick_test.py`:

- ROOT_PATH: "./results/mnist_quick_test/"
- EXP_PATH: "mnist_pgd_untargeted_linf_0.3"

Thus, the following command gives the corresponding performances:

```
python ./report.py ./results/mnist_quick_test/mnist_pgd_untargeted_linf_0.3/
```
# Training own model
You can train own models.
```
# On the project root path
cd arclu
```
A) Training a MNIST model
```
bash mnist_train_script.sh
```
B) Training a LightImageNet model
```
bash light_imagenet_train_script.sh
```
The training process generates model parameter files in `checkpoints` directory.

To load and test your models, you should change `test.py`.
```python
def test(args):
    __train_loader, test_loader, _ = load_data(args)
    if args.dataset == "mnist":
        y_mu_encoding = get_y_mu_encoding(M=10)
        model = MNISTModel(y_mu_encoding)
        model.load_state_dict(torch.load(
            "./checkpoints/mnist_2020_10_08_03_46_22/mnist_best_41.pth")) # <-- modify the string to your MNIST model path.
    elif args.dataset == "light_imagenet":
        y_mu_encoding = get_y_mu_encoding(args.NUM_CLASSES, 1000, 10)
        model = OursDataParallel(LightImageNetModel(y_mu_encoding, num_classes=args.NUM_CLASSES),
                                 device_ids=range(torch.cuda.device_count()))
        if args.ckpt == None:
            model.load_state_dict(torch.load(
                "./checkpoints/light_imagenet_2021_03_02_16_14_15/light_imagenet_160_999_0.385.pth")) # <-- modify the string to your LightImageNet model path.
        else:
            model.load_state_dict(torch.load(args.ckpt))
        model = model.module
    else:
        raise NotImplementedError()
```
<<<<<<< HEAD

# Environment detail
| Environment        	| version 	|
|------------	|------------	|
| GPU        	| GTX 1080ti 	|
| GPU driver 	| 418.152.00 	|
| CUDA       	| 10.1       	|
| python     	| 3.7.7      	|
| pytorch     	| 1.6         	|
| pytorchvision | 0.6.1         |
  
=======
>>>>>>> da7d827845c86ef1568d5f7b77f1b16f0c0a0502
