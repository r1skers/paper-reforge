# LeNet

Study folder for the LeNet / early CNN reading and reproduction thread.

## Paper

- **Title:** Gradient-Based Learning Applied to Document Recognition
- **Authors:** Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner
- **Venue:** Proceedings of the IEEE, 1998
- **DOI:** 10.1109/5.726791
- **PDF:** `papers/lecun-1998-gradient-based-learning-document-recognition.pdf`
- **Source:** http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

## Reproduction Target

This folder contains two LeNet-5 style MNIST reproduction paths:

- `src/train_paper.py`: paper-like version kept close to the 1998 architecture.
- `src/train.py`: modern CNN version using the same shape progression.

Both paths target:

1. reproduce a LeNet-5 style CNN on MNIST,
2. record every major layer shape,
3. train for 10-20 epochs,
4. save loss and accuracy.

Both implementations keep the LeNet layer shape pattern:

```text
input 1x32x32
-> C1 6x28x28
-> S2 6x14x14
-> C3 16x10x10
-> S4 16x5x5
-> C5 120x1x1
-> F6 84
-> output 10
```

## Paper-Like Version

Run:

```powershell
python .\src\train_paper.py --data-root ..\Variational_AutoEncoder\data --epochs 10 --output-dir outputs\paper_like
```

This version uses the paper-like details:

- scaled tanh in hidden convolutional / fully connected layers,
- trainable S2/S4 subsampling,
- C3 partial connection table,
- Euclidean RBF output distances.

Hidden layers use the LeNet scaled tanh:

```text
f(x) = 1.7159 * tanh(2x / 3)
```

The S2/S4 layers use paper-style trainable subsampling:

```text
2x2 region -> sum -> trainable alpha -> trainable bias -> sigmoid
```

C3 uses the paper's partial connection table from the 6 S2 maps to the 16 C3 maps.

The output layer is a Euclidean RBF head:

```text
F6 84-D vector -> squared distance to 10 fixed 84-D digit prototypes
```

Training uses `CrossEntropyLoss` over negative distances for stability:

```text
smaller distance -> larger class score
```

## Modern Version

Run:

```powershell
python .\src\train.py --data-root ..\Variational_AutoEncoder\data --epochs 10 --output-dir outputs\modern
```

This version keeps the LeNet shape progression but uses common modern CNN choices:

- full channel connectivity in C3,
- ReLU activations,
- max pooling for S2/S4,
- linear logits with `CrossEntropyLoss`.

## Setup

```powershell
cd D:\Dev\repos\paper-reforge\LeNet
pip install -r requirements.txt
```

## Train

Default modern training, 10 epochs:

```powershell
python .\src\train.py
```

Modern 20 epochs:

```powershell
python .\src\train.py --epochs 20
```

Reuse the MNIST data already downloaded by the VAE project:

```powershell
python .\src\train.py --data-root ..\Variational_AutoEncoder\data --epochs 10
```

## Outputs

Default output directory:

```text
outputs/run_lenet5_style/
```

Files:

- `config.json`
- `layer_shapes.csv`
- `metrics.csv`
- `training_curves.png`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
