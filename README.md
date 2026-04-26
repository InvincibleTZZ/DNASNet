# DNASNet
![](image/framework.png)
**Figure:** **A.**  Neural circuit evolution algorithm framework based on STDP rules

![](image/cluster_k_bilinear.png)
**Figure:** **B.** Example of a neuron cluster containing only one excitatory neuron and one inhibitory neuron, performing a secondary integration operation on the additive behavior of the neuron cluster

![](image/figure1.png)
**Figure:** **C.** The effect of bilinear coefficients was verified under a model with 8 Cell layers, 4 time steps, and 10M parameters, and it was found that the performance was improved on both static data sets and neuromorphic data sets. What’s interesting is that all initial values ​​of k_bilinear are set to 0.1. After training, kEE is stable around 0.2, kII is stable at 0.01fujin, and KEI is stable around 0.25.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

### CIFAR10

To train the models on CIFAR10 , run the following command:

```train
python NeuEvo_main_new.py --model NetworkCIFAR --dataset cifar10 --batch-size 128 --step 4 --layers 8 --arch cifar_new0
```
### DVS-CIFAR10
To train the models on DVS-CIFAR10 , run the following command:

```train
python NeuEvo_main_new.py --model NetworkCIFAR --dataset dvsc10 --batch-size 128 --step 4 --layers 8 --arch dvsc10_new1
```
### DVS-Gesture
To train the models on DVS-G , run the following command:

```train
python NeuEvo_main_new.py --model NetworkCIFAR --dataset dvsg --batch-size 128 --step 4 --layers 8 --arch dvsg_new2
```

