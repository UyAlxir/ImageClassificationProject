# 多种神经网络在多种图像分类任务上的性能对比以及对神经网络的优化

## 目录介绍

```txt
│  README.md
│
├─code				# 代码包
│  models.py 	 	 	# 三个神经网络模型
│  run.py		 	# 神经网络模型的测试
│  TraditionalModels.py 	# 传统模型，SVM和DecisionTree的测试
│
├─datasets  			# 数据集
│  ├─cifar-10-batches-py
│  ├─cifar-100-python
│  ├─FashionMNIST
│  └─MNIST
│
├─log 				# 模型运行产生的日志文件（最优的部分）
│      Conv2Layer_CIFAR100_02-18-20-24-25.txt
│      Conv2Layer_CIFAR10_02-18-14-22-00.txt
│      Conv2Layer_FMNIST_02-18-19-16-41.txt
│      Conv2Layer_MNIST_02-18-19-16-54.txt
│      DecisionTree_CIFAR100_02-18-20-32-55.txt
│      DecisionTree_CIFAR10_02-18-18-32-50.txt
│      DecisionTree_FMNIST_02-18-18-31-37.txt
│      DecisionTree_MNIST_02-18-18-30-49.txt
│      FC2Layer_CIFAR100_02-18-20-23-36.txt
│      FC2Layer_CIFAR10_02-18-19-32-51.txt
│      FC2Layer_FMNIST_02-18-20-06-01.txt
│      FC2Layer_MNIST_02-18-20-14-34.txt
│      FinalNet_CIFAR100_02-19-09-57-13.txt
│      FinalNet_CIFAR10_02-19-10-48-29.txt
│      FinalNet_FMNIST_02-19-10-12-15.txt
│      FinalNet_MNIST_02-19-10-29-50.txt
│      SVM_CIFAR100_02-18-21-49-30.txt
│      SVM_CIFAR10_02-18-16-53-43.txt
│      SVM_FMNIST_02-18-15-46-38.txt
│      SVM_MNIST_02-18-15-45-06.txt
│
└─saved_model			# 三个神经网络模型，保存的最优的模型
    ├─Conv2Layer
    │      CIFAR10.bin
    │      CIFAR100.bin
    │      FMNIST.bin
    │      MNIST.bin
    │
    ├─FC2Layer
    │      CIFAR10.bin
    │      CIFAR100.bin
    │      FMNIST.bin
    │      MNIST.bin
    │
    └─FinalNet
            CIFAR10.bin
            CIFAR100.bin
            FMNIST.bin
            MNIST.bin
```

## 神经网络

### train

```cmd
cd code
python run.py ^
      --do_train ^
      --do_test ^
      --model FC2Layer ^
      --model_path ../saved_model/ ^
      --dataset FMNIST ^
      --train_batch_size 64 ^
      --test_batch_size 64 ^
      --with_cuda ^
      --lr 0.99 ^
      --epoch 20
```

- 模型

> 以上指令中训练的是FC2Layer模型，还支持Conv2Layer和FinalNet，只需将model参数改为相应的模型即可

- 数据集

> 以上为MNIST数据集示例，还可支持FMNIST，CIFAR10和CIFAR100，只需将
> dataset 参数改为相应的数据集即可

- GPU

> 通过使用 with_cuda 参数可以使用GPU

### test

```cmd
cd code
python run.py ^
      --do_test ^
      --do_valid ^
      --model FC2Layer ^
      --model_path ../saved_model/ ^
      --dataset MNIST ^
      --test_batch_size 64 ^
      --with_cuda
```

## SVM

```cmd
python TraditionalModels.py ^
    --model SVM ^
    --dataset CIFAR100 ^
    --datasets_path ../datasets/ ^
    --log_path ../log/ ^
    --max_iter 5
  
```

## DecisionTree

```cmd
python TraditionalModels.py ^
    --model DecisionTree ^
    --dataset CIFAR100 ^
    --datasets_path ../datasets/ ^
    --log_path ../log/

```
