# TokenFormer with P-Attention Layer: README

## Overview

This project integrates the **P-Attention Layer** into the TokenFormer architecture, inspired by Andrej Karpathy's [NG Video Lecture project](https://github.com/karpathy/ng-video-lecture) and the [TokenFormer paper](https://arxiv.org/pdf/2410.23168). The P-Attention Layer enhances token-level representation by incorporating positional relationships within sequences, aiming to improve performance in sequence modeling tasks.

---

## Features

- **P-Attention Integration**: Adds a novel Positional Attention mechanism to TokenFormer, improving its capacity for sequential modeling.
- **Foundation in TokenFormer**: Builds on the core principles of the TokenFormer architecture for modularity and flexibility.
- **Inspired by Karpathy's Simplicity**: The implementation mirrors the modular and educational style of the NG Video Lecture repository.
- **Community Testing Encouraged**: Provides a framework for exploration and invites the community to extend its capabilities.

---

## References

1. **TokenFormer Paper**: [TokenFormer: Enhancing Sequence Models with Token-Level Representations](https://arxiv.org/pdf/2410.23168)  
2. **Karpathy's NG Video Lecture**: [GitHub Repository](https://github.com/karpathy/ng-video-lecture)

---

## Acknowledgments

This project would not have been possible without the insights provided by Andrej Karpathy’s work and the innovative concepts introduced in the TokenFormer paper. Their contributions laid the foundation for this implementation.

---

### Note to the Community

While the core functionality of the model and the P-Attention Layer has been implemented, **the capability of training the model by incrementing the Key and Value parameters sequentially in the P-Attention Layer has not yet been tested**.  

If you are interested in exploring this, consider running experiments on GPUs and sharing your findings with the community!
