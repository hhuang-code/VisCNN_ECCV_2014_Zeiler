# VisCNN_ECCV_2014_Zeiler

## Brief Introduction
```
A PyTorch implementation of the paper "Visualizing and understanding convolutional networks" and rely on https://github.com/csgwon/pytorch-deconvnet
```

## Modifications
```
1. add ReLU layers to deconvolutional network
2. delete "deconv_first_layers"
3. use hooks to get intermediate feature maps
4. add heatmap
5. other modifications to make the code more elegant
```

## Paper
```
@inproceedings{zeiler2014visualizing,
  title={Visualizing and understanding convolutional networks},
  author={Zeiler, Matthew D and Fergus, Rob},
  booktitle={European conference on computer vision},
  pages={818--833},
  year={2014},
  organization={Springer}
}
```
