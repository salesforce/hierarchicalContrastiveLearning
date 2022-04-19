# hierarchicalContrastiveLearning
This repo covers an reference implementation for the follwing paper in PyTorch, using Deep Fashion In-Store as an illustrative example:
Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework.

## Requirements
* Pytorch 1.7.0
* tensorboard_logger 0.1.0

## Running
* This code is built upon two codebases: [Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast) and [MoCo](https://github.com/facebookresearch/moco).
* Train pre-trained model on Deep Fashion In-store dataset
	* Perpare train-listfile, val-listfile
```
python train_deepfashion.py --data ./deepfashion/ --train-listfile ./train_listfile.json --val-listfile ./val_listfile.json --class-map-file ./classmap.json --num-classes 17 --feature-extract --learning_rate 0.9 --temp 0.1

```

## Reference
```
@inproceedings{hierarchicalContrastiveLearning,
      title={Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework}, 
      author={Shu Zhang and Ran Xu and Caiming Xiong and Chetan Ramaiah},
      year={2022},
      booktitle={CVPR},
}

```