# FedEntropy: Efficient Federated Learning for Non-IID Scenarios Using Maximum Entropy Judgment-based Device Selection

This is the code for paper FedEntropy: Efficient Federated Learning for Non-IID Scenarios Using Maximum Entropy Judgment-based Device Selection.

Abstract: 
Federated Learning (FL) has attracted steadily increasing attention as a promising distributed machine learning paradigm, which enables the training of a globally 
centralized model for numerous decentralized devices (i.e., clients)
 without exposing their privacy. 
However, due to the biased data distributions on involved clients, FL inherently suffers from low classification accuracy in non-IID  (Independently Identically Distribution) scenarios. 
Although various device grouping methods have been proposed to address this problem, most of them neglect both 
i) distinct data distribution  characteristics of heterogeneous devices, and
ii) contributions and hazards of local models, 
which are extremely important in determining the quality of global model aggregation. 
In this paper, we present an effective FL approach, named FedEntropy, with a novel two-stage
dynamic device selection scheme, which makes full use of the two factors above based on our proposed maximum entropy judgement heuristics. 
Unlike existing FL methods that directly and equally
aggregate local models collected
from all the selected clients, FedEntropy first
selects devices with high potential for benefiting global model aggregation in a coarse manner, and then further filters out inferior devices from such selected devices based on 
our proposed maximum entropy judgement method.
 Based on the pre-collected soft labels of the selected devices, FedEntropy
 only aggregates those local models that can maximize the overall entropy of their soft labels. This way, without collecting local models that are malicious for the aggregation, FedEntropy can effectively improve global model accuracy while reducing the overall communication overhead. 
Comprehensive  experimental results on well-known benchmarks show that FedEntropy not only outperforms state-of-the-art FL methods in terms of the model accuracy and communication overhead, but also can be integrated into these methods to enhance their classification performance.  

## Dependencies
* PyTorch >= 1.8.0
* torchvision >= 0.9.0

## Parameters
|  Parameter | Description  |
|  ----  | ----  |
| model  | The model architecture. Options: LeNet, MobileNet, VGG, ResNet |
| dataset  | Dataset to use. Options: CIFAR-10. CIFAR-100, CINIC-10 |
| epochs  | Number of communication rounds |
| num_users  | Number of users |
| frac  | the fraction of users to be sampled in each round |
| local_ep  | Number of local epochs |
| bs  | Test batch size |
| local_bs  | Local train batch size |
| lr  | Learning rate |
| algorithm  | Federated Learning algorithms. Options: FedAvg, FedProx, Moon, FedEntropy |
| rule  | Non-IID case. Options: Drichlet, ill |
| Drichlet_arg  | The concentration parameter of the Dirichlet distribution for non-IID partition. |
| ill_case  | The concentration parameter of the ill distribution for non-IID partition. Options: 1, 2|
| num_classes  | Number of classes of image |
| num_channels  | Number of channels of image |
| threshold  | The threshold of epsilon-policy|


## Usage
Here is an example to run FedEntropy on CIFAR-10 with LeNet:  
case 1:
```bash
python main.py --algorithm FedEntropy --dataset cifar10 --model LeNet --local_bs 50 --rule ill --ill_case 1
```
case 2:
```bash
python main.py --algorithm FedEntropy --dataset cifar10 --model LeNet --rule ill --ill_case 2
```
case 3:
```bash
python main.py --algorithm FedEntropy --dataset cifar10 --model LeNet --rule Drichlet --Drichlet_arg 0.1
```

## Hyperparameters
If you use the same setting as our papers, you can simply adopt the hyperparameters reported in our paper. 
