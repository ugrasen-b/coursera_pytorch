# PyTorch Learning & Experiments

My code, notes and experiments from working through Coursera PyTorch courses. This repo tracks my progress, from tensors and simple linear models to more advanced deep learning.

## Repository Structure

| File | Description |
|------|-------------|
| `first_pytorch.py` | First PyTorch model: a single-neuron linear regression (`nn.Linear(1, 1)`) that predicts delivery time from distance, trained with MSE loss and SGD. Also tries out basic tensor operations (`unsqueeze`, `squeeze`, broadcasting). |
| `helper_utils.py` | Plotting helpers built on Matplotlib: `plot_results` (actual vs. predicted values) and `plot_nonlinear_comparison` (how the model does on non-linear data). |
| `test.py` | Scratch file for quick tests. |
| `environment.yml` | Conda environment with every dependency used in this repo. |

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd pytorch
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate pytorch-env
```

This sets up Python 3.13 with:

- **PyTorch stack:** `torch`, `torchvision`, `torchaudio`, `torchmetrics`, `torchinfo`
- **Data science:** `numpy`, `pandas`, `scipy`, `scikit-learn`
- **Visualization & tracking:** `matplotlib`, `seaborn`, `tensorboard`
- **Utilities:** `tqdm`, `pillow`, `requests`
- **Notebooks:** `jupyter`, `ipykernel`

> **GPU support:** `pip install torch` installs the default build. For a specific CUDA version, follow the instructions at [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 3. Run an example

```bash
python first_pytorch.py
```

It prints the training loss every 10 epochs, then the output of a few tensor operations.

## Topics Covered

- [x] Tensors and basic tensor operations
- [x] Building a model with `nn.Sequential` and `nn.Linear`
- [x] Loss functions (`nn.MSELoss`) and optimizers (`optim.SGD`)
- [x] The training loop: `zero_grad` → forward → loss → `backward` → `step`
- [x] Plotting predictions with Matplotlib
- [ ] Non-linear models and activation functions
- [ ] Datasets and DataLoaders
- [ ] Convolutional neural networks (CNNs)
- [ ] Transfer learning with `torchvision`

*(I update this checklist as I go through the course.)*

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Coursera](https://www.coursera.org/)

## Note

This repo is for personal learning. Some helper code may come from course materials and belongs to its original authors.
