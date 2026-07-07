# Connecting ResNets and Neural ODEs

*Continuous-time perspective on deep residual learning*

This repository is provided alongside an exploratory notebook on how deep residual networks relate to neural ordinary differential equations. The central thread is geometric: a ResNet with a given step size is treated as a forward Euler discretization of continuous dynamics. When depth is pushed to the limit, a neural ODE is obtained.

## What you will find

The notebook `code.ipynb` is organized into four movements.

First, a deep feed-forward network and backpropagation are introduced, and the vanishing-gradient problem on Two-Moons and MNIST is demonstrated by plotting gradient norms across layers.

Second, ResNet skip connections are introduced and the same gradient analysis is repeated, illustrating how identity shortcuts preserve gradient flow in very deep models.

Third, plain and residual architectures are compared on Two-Moons classification.

Fourth, the continuous-time limit is explored: ResNet updates are cast as an ODE, implemented with forward Euler and RK4 integrators, trained on Two-Moons, and analyzed for integrator accuracy.

## Running the notebook

Python 3 with PyTorch, NumPy, Matplotlib, scikit-learn, and torchvision (for MNIST) is required. `code.ipynb` should be opened in Jupyter and all cells run from top to bottom. CUDA is selected automatically when available.

## Main takeaway

Depth can be viewed either as a stack of discrete layers or as evolution in a continuous "time" variable. ResNets sit between these views: they are characterized as deep discrete models whose structure mirrors a continuous dynamical system. In neural ODEs, that connection is made explicit.

## References

-  He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770–778.
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *Advances in Neural Information Processing Systems*, 31, 6571–6583.
- Haber, E., & Ruthotto, L. (2017). Stable architectures for deep neural networks. *Inverse Problems*, 34(1), 014004.
- E, W. (2017). A proposal on machine learning via dynamical systems. *Communications in Mathematics and Statistics*, 5(1), 1–11.
- Kidger, P. (2022). *On neural differential equations* (Doctoral dissertation, University of Oxford). arXiv preprint arXiv:2202.02435.
