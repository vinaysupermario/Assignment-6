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
   1.  Your Repo link where we can see your test logs (to see test accuracy) IN THE README.MD section: this file
   2.  Your GitHub Actions Link: 
   3.  Your Test file link: 
   4.  Your Validation Test Accuracy: 99.50 (18th epoch)
7.  Result:
    1. Parameters: 12.1k
    2. Best Train Accuracy: 99.23
    3. Best Test Accuracy: 99.50
    4. Epochs: 20

## Training Logs

```
EPOCH: 0
Loss=0.07571855187416077 Batch_id=468 Accuracy=90.74: 100%|██████████████████████████| 469/469 [00:15<00:00, 30.07it/s]

Test set: Average loss: 0.0548, Accuracy: 9827/10000 (98.27%)

EPOCH: 1
Loss=0.0382654182612896 Batch_id=468 Accuracy=97.98: 100%|███████████████████████████| 469/469 [00:15<00:00, 31.24it/s]

Test set: Average loss: 0.0349, Accuracy: 9892/10000 (98.92%)

EPOCH: 2
Loss=0.022863052785396576 Batch_id=468 Accuracy=98.35: 100%|█████████████████████████| 469/469 [00:17<00:00, 27.26it/s]

Test set: Average loss: 0.0310, Accuracy: 9903/10000 (99.03%)

EPOCH: 3
Loss=0.05257285013794899 Batch_id=468 Accuracy=98.58: 100%|██████████████████████████| 469/469 [00:15<00:00, 29.43it/s]

Test set: Average loss: 0.0260, Accuracy: 9919/10000 (99.19%)

EPOCH: 4
Loss=0.03009389340877533 Batch_id=468 Accuracy=98.70: 100%|██████████████████████████| 469/469 [00:14<00:00, 32.32it/s]

Test set: Average loss: 0.0239, Accuracy: 9926/10000 (99.26%)

EPOCH: 5
Loss=0.05914641544222832 Batch_id=468 Accuracy=98.75: 100%|██████████████████████████| 469/469 [00:15<00:00, 30.42it/s]

Test set: Average loss: 0.0246, Accuracy: 9928/10000 (99.28%)

EPOCH: 6
Loss=0.0335736945271492 Batch_id=468 Accuracy=98.85: 100%|███████████████████████████| 469/469 [00:14<00:00, 31.50it/s]

Test set: Average loss: 0.0210, Accuracy: 9929/10000 (99.29%)

EPOCH: 7
Loss=0.04438810423016548 Batch_id=468 Accuracy=98.94: 100%|██████████████████████████| 469/469 [00:14<00:00, 31.49it/s]

Test set: Average loss: 0.0242, Accuracy: 9935/10000 (99.35%)

EPOCH: 8
Loss=0.034740615636110306 Batch_id=468 Accuracy=98.95: 100%|█████████████████████████| 469/469 [00:14<00:00, 31.34it/s]

Test set: Average loss: 0.0216, Accuracy: 9936/10000 (99.36%)

EPOCH: 9
Loss=0.003565046703442931 Batch_id=468 Accuracy=99.01: 100%|█████████████████████████| 469/469 [00:14<00:00, 31.88it/s]

Test set: Average loss: 0.0217, Accuracy: 9933/10000 (99.33%)

EPOCH: 10
Loss=0.016259953379631042 Batch_id=468 Accuracy=98.97: 100%|█████████████████████████| 469/469 [00:14<00:00, 31.36it/s]

Test set: Average loss: 0.0201, Accuracy: 9936/10000 (99.36%)

EPOCH: 11
Loss=0.011176113970577717 Batch_id=468 Accuracy=99.07: 100%|█████████████████████████| 469/469 [00:14<00:00, 32.09it/s]

Test set: Average loss: 0.0195, Accuracy: 9940/10000 (99.40%)

EPOCH: 12
Loss=0.030532604083418846 Batch_id=468 Accuracy=99.09: 100%|█████████████████████████| 469/469 [00:14<00:00, 32.66it/s]

Test set: Average loss: 0.0195, Accuracy: 9937/10000 (99.37%)

EPOCH: 13
Loss=0.0898468866944313 Batch_id=468 Accuracy=99.11: 100%|███████████████████████████| 469/469 [00:15<00:00, 29.61it/s]

Test set: Average loss: 0.0187, Accuracy: 9949/10000 (99.49%)

EPOCH: 14
Loss=0.013013019226491451 Batch_id=468 Accuracy=99.14: 100%|█████████████████████████| 469/469 [00:14<00:00, 32.25it/s]

Test set: Average loss: 0.0189, Accuracy: 9941/10000 (99.41%)

EPOCH: 15
Loss=0.022816047072410583 Batch_id=468 Accuracy=99.18: 100%|█████████████████████████| 469/469 [00:14<00:00, 32.56it/s]

Test set: Average loss: 0.0196, Accuracy: 9943/10000 (99.43%)

EPOCH: 16
Loss=0.0058556124567985535 Batch_id=468 Accuracy=99.15: 100%|████████████████████████| 469/469 [00:15<00:00, 29.38it/s]

Test set: Average loss: 0.0195, Accuracy: 9937/10000 (99.37%)

EPOCH: 17
Loss=0.005618322640657425 Batch_id=468 Accuracy=99.13: 100%|█████████████████████████| 469/469 [00:14<00:00, 31.42it/s]

Test set: Average loss: 0.0170, Accuracy: 9947/10000 (99.47%)

EPOCH: 18
Loss=0.04472185671329498 Batch_id=468 Accuracy=99.23: 100%|██████████████████████████| 469/469 [00:15<00:00, 30.18it/s]

Test set: Average loss: 0.0186, Accuracy: 9950/10000 (99.50%)

EPOCH: 19
Loss=0.0014880821108818054 Batch_id=468 Accuracy=99.20: 100%|████████████████████████| 469/469 [00:14<00:00, 32.30it/s]

Test set: Average loss: 0.0177, Accuracy: 9942/10000 (99.42%)
```