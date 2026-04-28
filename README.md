# DNASNet

![](image/pipeline.png)
**Figure:** **A.**  This figure illustrates the complete pipeline of DNASNet, spanning from input encoding and decoupled neural architecture search to the final retraining and prediction. Initially, static images or event data are encoded into spike trains across time steps and fed into a supernet composed of multiple searchable Cells. During the search phase, DNASNet fundamentally decouples the coupled bilevel optimization prevalent in traditional SNN-NAS. Specifically, it employs global Spatio-Temporal Backpropagation (STBP) to update the synaptic weights $w$ based on the training loss, while independently updating the architecture parameters $\alpha$ via Spike-Timing-Dependent Plasticity (STDP) governed by local spike timing during the validation forward pass. This explicitly avoids the prohibitive computational overhead of gradient backpropagation through architecture variables. As the search iterates, competition arises among candidate operations within the Cell; operations exhibiting stronger temporal causality acquire higher weights. Ultimately, a winner-takes-all strategy is executed to retain the operation with the highest weight, deriving a sparse and discrete Cell architecture. In the retraining phase, these searched Cells are stacked to construct the final SNN. Crucially, a biologically-inspired dendritic bilinear integration module is introduced at the selected dual-branch nodes to model the non-linear interplay between excitatory and inhibitory pathways via $O^t = Y_1^t + Y_2^t + k(Y_1^t \odot Y_2^t)$. Finally, the network temporally averages the outputs across all time steps to yield the classification prediction. Overall, this pipeline highlights the two core innovations of DNASNet: the highly efficient STDP-driven decoupled architecture search, and the dendritic bilinear integration mechanism that significantly boosts representational capacity during retraining.


## Requirements

To install requirements:

```setup
Python >= 3.9.25, Pytoch == 2.7.1, torchvision == 0.22.1
```
## Datasets
CIFAR-10, CIFAR-100 can be automatically downloaded via `torchvision`.
Here is the download address of the neuromorphic datasets [CIFAR10-DVS](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671) and [DVS128Gesture](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794).

## Training

### CIFAR10

To train the models on CIFAR10 , run the following command:

```train
CUDA_VISIBLE_DEVICES=0 python DNASNet_train.py --model NetworkCIFAR --dataset cifar10 --batch-size 128 --step 4 --layers 16 --node-type PLIFNode --init-channels 36 --arch cifar_final --lr 0.005 --epochs 600 --use-bilinear true
```
### DVS-CIFAR10
To train the models on DVS-CIFAR10 , run the following command:

```train
python DNASNet_train.py --model NetworkCIFAR --dataset dvsc10 --batch-size 64 --step 10 --layers 16 --arch dvsc10_base3 --node-type PLIFNode --init-channels 36 --lr 0.005 --epochs 600 --use-bilinear true
```
### DVS-Gesture
To train the models on DVS-G , run the following command:

```train
python DNASNet_train.py --model NetworkCIFAR --dataset dvsg --batch-size 64 --step 10 --layers 16 --arch dvsg_new2 --node-type PLIFNode --init-channels 36 --lr 0.005 --epochs 600 --use-bilinear true
```
## Search
### Search on CIFAR10
To search the Cell on CIFAR10 , run the following command:

```train
CUDA_VISIBLE_DEVICES=0 python train_search.py --dataset cifar10 --epochs 50 --batch-size 128 --init-channels 16 --layers 6 --step 4 --stdp-type full --learning_rate 0.005
```

### Search on DVS-CIFAR10
To search the Cell on DVS-CIFAR10 , run the following command:

```train
CUDA_VISIBLE_DEVICES=0 python train_search.py --dataset dvsc10 --epochs 50 --batch-size 64 --init-channels 16 --layers 6 --step 10 --stdp-type full --learning_rate 0.005
```
