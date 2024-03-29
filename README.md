# Baseline Experiments: Polynomial-Based Graph Convolutional Neural Networks For Graph Classification

Graph convolutional neural networks exploit convolution operators, based on some neighborhood aggregating scheme, to compute representations of graphs. The most common convolution operators only exploit local topological information. To consider wider topological receptive fields, the mainstream approach is to non-linearly stack multiple graph convolutional (GC) layers. In this way, however, interactions among GC parameters at different levels pose a bias on the flow of topological information. In this paper, we propose a different strategy, considering a single graph convolution layer that independently exploits neighbouring nodes at different topological distances, generating decoupled representations for each of them. These representations are then processed by subsequent readout layers. We implement this strategy introducing the polynomial graph convolution (PGC) layer, that we prove being more expressive than the most common convolution operators and their linear stacking. Our contribution is not limited to the definition of a convolution operator with a larger receptive field, but we prove both theoretically and experimentally that the common way multiple non-linear graph convolutions are stacked limits the neural network expressiveness. Specifically, we show that a graph neural network architecture with a single PGC layer achieves state of the art performance on many commonly adopted graph classification benchmarks.

Paper: https://link.springer.com/article/10.1007/s10994-021-06098-0

If you find this code useful, please cite the following:

> @article{pasa2021polynomial,  
   title={Polynomial-based graph convolutional neural networks for graph classification},  
   author={Pasa, Luca and Navarin, Nicol{\`o} and Sperduti, Alessandro},   
   journal={Machine Learning},  
   pages={1--33},  
   year={2021},  
   publisher={Springer}   
}



