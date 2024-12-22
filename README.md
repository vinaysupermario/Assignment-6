# PerfectNet : MNIST 

## The pipeline will:

1. Train the model for 1 epochs
2. Run tests checking:
    - Model architecture (input shape 28x28, output shape 10)
    - Parameter count (<= 20000)
    - Use of Batch Normalization
    - Use of DropOut
    - Use of Fully Connected Layer or GAP
    - Model accuracy (> 95% on first test set)
3. Do not save the model using Github Actions, because to properly train it we need 20 epochs. So we train it manually and upload the model manually over to github.

## The model architecture:
Our Convolutional Neural Network uses:
- 9 Convolutional blocks with:
  - Conv2d layers (varying channels: 1→8→16→8→16→16→16→16→16→10)
  - ReLU activation
  - Batch Normalization
  - Dropout (0.05)
- 1 MaxPool2d layer
- Global Average Pooling (GAP)
- Log Softmax output

Key features:
- Uses 1x1 convolutions for channel reduction
- Employs skip connections
- Consistent dropout rate of 0.05
- Total parameters: ~12.1k

## Target:
1. Less than 20k parameters ✓
2. Less than 20 epochs ✓
3. To use Batch Normalization and Dropout ✓
4. To use a Fully Connected Layer or GAP ✓
5. Create a GitHub Actions file, so we can check:
   1.  **Total Parameter Count Test** ✓
   2.  **Total Epoch Count Test** ✓
   3.  **Use of Batch Normalization** ✓
   4.  **Use of DropOut** ✓
   5.  **Use of Fully Connected Layer or GAP** ✓
6. Submission Items:
   1.  Your Repo link where we can see your test logs (to see test accuracy) IN THE README.MD section: **Training Logs Down at the bottom of THIS FILE**
   2.  Your GitHub Actions Link: https://github.com/vinaysupermario/Assignment-6/actions/runs/12454558603/job/34765947698
   3.  Your Test file link: https://github.com/vinaysupermario/Assignment-6/blob/main/test_model.py
   4.  Your Validation Test Accuracy: 99.57 (19th epoch)
7.  Result:
    1. Parameters: 12.1k
    2. Best Train Accuracy: 99.23
    3. Best Test Accuracy: 99.40(11th epoch), 99.57(19th epoch)
    4. Epochs: 20

## Training Logs

```
CUDA Available? True
EPOCH: 0
Loss=0.07118798792362213 Batch_id=468 Accuracy=90.75: 100%|██████████████████████████| 469/469 [00:15<00:00, 30.26it/s]

Test set: Average loss: 0.0484, Accuracy: 9846/10000 (98.46%)

EPOCH: 1
Loss=0.032985854893922806 Batch_id=468 Accuracy=97.99: 100%|█████████████████████████| 469/469 [00:14<00:00, 31.34it/s]

Test set: Average loss: 0.0343, Accuracy: 9892/10000 (98.92%)

EPOCH: 2
Loss=0.02779148519039154 Batch_id=468 Accuracy=98.34: 100%|██████████████████████████| 469/469 [00:14<00:00, 31.58it/s]

Test set: Average loss: 0.0292, Accuracy: 9905/10000 (99.05%)

EPOCH: 3
Loss=0.0700560137629509 Batch_id=468 Accuracy=98.54: 100%|███████████████████████████| 469/469 [00:14<00:00, 31.44it/s]

Test set: Average loss: 0.0275, Accuracy: 9913/10000 (99.13%)

EPOCH: 4
Loss=0.05359397828578949 Batch_id=468 Accuracy=98.75: 100%|██████████████████████████| 469/469 [00:14<00:00, 31.34it/s]

Test set: Average loss: 0.0214, Accuracy: 9928/10000 (99.28%)

EPOCH: 5
Loss=0.05548921227455139 Batch_id=468 Accuracy=98.77: 100%|██████████████████████████| 469/469 [00:14<00:00, 31.27it/s]

Test set: Average loss: 0.0237, Accuracy: 9928/10000 (99.28%)

EPOCH: 6
Loss=0.05254469811916351 Batch_id=468 Accuracy=98.83: 100%|██████████████████████████| 469/469 [00:17<00:00, 26.19it/s]

Test set: Average loss: 0.0216, Accuracy: 9930/10000 (99.30%)

EPOCH: 7
Loss=0.05432368442416191 Batch_id=468 Accuracy=98.90: 100%|██████████████████████████| 469/469 [00:16<00:00, 28.47it/s]

Test set: Average loss: 0.0265, Accuracy: 9912/10000 (99.12%)

EPOCH: 8
Loss=0.01854737475514412 Batch_id=468 Accuracy=98.97: 100%|██████████████████████████| 469/469 [00:14<00:00, 31.39it/s]

Test set: Average loss: 0.0213, Accuracy: 9936/10000 (99.36%)

EPOCH: 9
Loss=0.007743133697658777 Batch_id=468 Accuracy=99.01: 100%|█████████████████████████| 469/469 [00:15<00:00, 31.18it/s]

Test set: Average loss: 0.0204, Accuracy: 9935/10000 (99.35%)

EPOCH: 10
Loss=0.03942243009805679 Batch_id=468 Accuracy=99.02: 100%|██████████████████████████| 469/469 [00:14<00:00, 31.49it/s]

Test set: Average loss: 0.0207, Accuracy: 9929/10000 (99.29%)

EPOCH: 11
Loss=0.01126155350357294 Batch_id=468 Accuracy=99.07: 100%|██████████████████████████| 469/469 [00:14<00:00, 31.54it/s]

Test set: Average loss: 0.0201, Accuracy: 9940/10000 (99.40%)

EPOCH: 12
Loss=0.026678336784243584 Batch_id=468 Accuracy=99.11: 100%|█████████████████████████| 469/469 [00:16<00:00, 28.83it/s]

Test set: Average loss: 0.0172, Accuracy: 9945/10000 (99.45%)

EPOCH: 13
Loss=0.09404268115758896 Batch_id=468 Accuracy=99.12: 100%|██████████████████████████| 469/469 [00:15<00:00, 30.80it/s]

Test set: Average loss: 0.0169, Accuracy: 9955/10000 (99.55%)

EPOCH: 14
Loss=0.008738379925489426 Batch_id=468 Accuracy=99.21: 100%|█████████████████████████| 469/469 [00:15<00:00, 31.22it/s]

Test set: Average loss: 0.0181, Accuracy: 9942/10000 (99.42%)

EPOCH: 15
Loss=0.01946774683892727 Batch_id=468 Accuracy=99.19: 100%|██████████████████████████| 469/469 [00:14<00:00, 31.44it/s]

Test set: Average loss: 0.0179, Accuracy: 9944/10000 (99.44%)

EPOCH: 16
Loss=0.00875325407832861 Batch_id=468 Accuracy=99.14: 100%|██████████████████████████| 469/469 [00:15<00:00, 31.27it/s]

Test set: Average loss: 0.0186, Accuracy: 9948/10000 (99.48%)

EPOCH: 17
Loss=0.01115016546100378 Batch_id=468 Accuracy=99.21: 100%|██████████████████████████| 469/469 [00:14<00:00, 31.27it/s]

Test set: Average loss: 0.0154, Accuracy: 9952/10000 (99.52%)

EPOCH: 18
Loss=0.047741565853357315 Batch_id=468 Accuracy=99.20: 100%|█████████████████████████| 469/469 [00:15<00:00, 30.76it/s]

Test set: Average loss: 0.0183, Accuracy: 9943/10000 (99.43%)

EPOCH: 19
Loss=0.0018635994056239724 Batch_id=468 Accuracy=99.19: 100%|████████████████████████| 469/469 [00:17<00:00, 27.12it/s]

Test set: Average loss: 0.0154, Accuracy: 9957/10000 (99.57%)

Model saved as models/model_20241222_183408.pth
```