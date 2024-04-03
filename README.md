# Unlearning Inversion Attacks

This repository contains Python code for the inversion attack in machine unlearning in the paper "Learn What You Want to Unlearn: Unlearning Inversion Attacks against Machine Unlearning".

The code has been tested under CUDA 11.8, Python 3.9.13 and Pytorch 2.0.1.

There are two components in our code: preparation of the pretrained model (for learning and unlearning later) and data recovery/label inference for unlearned model. 


## Part 1: Prepare the pretrained model
Run the following command (with ResNet18 and STL-10 as example) to prepare the 
```
python prepare_model.py --model ResNet18 --dataset stl10 --exclude_num 1000 --seed 0 --save_folder results/models
```
where `--exclude_num` is the leave-out samples for learning/unlearning later, the saved checkpoints are stored under `--save_folder`. The dataset are stored under `./datasets`.

## Part 2-1: Recover the unlearned samples
The script `recovery_data.py` automatically test the exact unlearning and the approximate unlearning. The command can be 
```
python recovery_data.py --model ResNet18 --dataset stl10 --ft_samples 1000 --unlearn_samples 1 --seed 0 --model_save_folder results/models
```
where `--ft_samples` is the size of finetuning dataset and `--unlearn_samples` controls the number of unlearned samples.

## Part 2-2: Infer the unlearned labels

There are two steps. First, we use `gen_probing_samples.py` to generate probing samples for a given model and for each class.
For example, to generate probing samples for class 0, we run
```
python gen_probing_samples.py --classid 0 --model_save_folder results/models --redo_ft
```
where `--redo_ft` enables re-finetuning the pretrained model. We can disable it to save time if there is already a finetuned one.


Second, we use the probing samples (after manually combining the probing samples from all class into a pickled dictionary named `query_sample_dict.pkl` under the `probing_sample` subfolder under the model folder) and `label_inference.py` to infer the unlearned label for the exact unlearning and the approximate unlearning. The example command can be

```
python label_inference.py --model_save_folder results/models --load_folder_name resnet18_stl10_ex1000_s0
```
