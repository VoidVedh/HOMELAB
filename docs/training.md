# Training Protocol & Methodology

## Overview
This document specifies the training methodology, loss functions, optimizer configurations, and learning rate schedules used across the `ai_core` framework.

## Training Pipeline

### Loss Function
- **Cross-Entropy Loss**: Computed over token logit predictions against shifted target tokens.
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid y_{<i})$$

### Optimizer & Regularization
- **AdamW Optimizer**:
  - $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$
  - Decoupled weight decay applied exclusively to 2D matrix weights (embeddings & linear projections), excluding 1D bias and layer norm parameters.
  - Weight Decay: $0.1$
  - Gradient Clipping: Maximum norm $1.0$

### Learning Rate Schedule
- **Cosine Annealing with Warmup**:
  - Warmup Steps: First 5% of total training steps (linear ramp from 0 to $\eta_{max}$).
  - Cosine Decay: Decay from $\eta_{max}$ down to $\eta_{min} = 0.1 \times \eta_{max}$.

```python
# Conceptual Learning Rate Curve
#      /\
#     /  \
#    /    \___
#   0   Warmup  Total Steps
```

## Checkpoint Strategy
- Checkpoints saved to `checkpoints/<model_name>/` containing:
  - `model_state_dict`
  - `optimizer_state_dict`
  - `epoch` & `global_step`
  - `val_loss` & `perplexity`
  - `model_config` dict
