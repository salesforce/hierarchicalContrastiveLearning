# Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework
This repo covers an reference implementation for the paper [Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework](https://arxiv.org/abs/2204.13207) in PyTorch, using Deep Fashion In-Store as an illustrative example:
Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework.

## Requirements
* Pytorch 1.7.0
* tensorboard_logger 0.1.0

## Typo
* We want to correct a typo in the paper. 
In Eq. 6 and 7, <img src="https://render.githubusercontent.com/render/math?math=l%2d%201"> should be <img src="https://render.githubusercontent.com/render/math?math=l%2b%201">.

## Running
* This code is built upon two codebases: [Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast) and [MoCo](https://github.com/facebookresearch/moco).
* Train pre-trained model on Deep Fashion In-store dataset
	* Perpare train-listfile, val-listfile. The format is as follows:
	```
	{
	  "images": [
	    "/deep_fashion_in_store/img/WOMEN/Dresses/id_00000002/02_1_front.jpg",
	    "/deep_fashion_in_store/img/WOMEN/Dresses/id_00000002/02_2_side.jpg",
	    "/deep_fashion_in_store/img/WOMEN/Dresses/id_00000002/02_4_full.jpg",
	    "/deep_fashion_in_store/img/WOMEN/Dresses/id_00000002/02_7_additional.jpg",
	    "/deep_fashion_in_store/img/WOMEN/Blouses_Shirts/id_00000004/03_1_front.jpg"
	  ],
	  "categories": [
	    "Dresses",
	    "Dresses",
	    "Dresses",
	    "Dresses",
	    "Blouses_Shirts"
	  ]
	}

	```
	
	* Class map can be downloaded from [class_map.json](https://drive.google.com/file/d/19q9NnnCieycgfsLI-iQCdTu82oDRZTAO/view)
	* Repeating product ids can be downloaded from [repeating_product_ids.csv](https://drive.google.com/file/d/1oFZfmZNTQNkPOiyIc_4b_g3qIXDvTmv4/view?usp=sharing)
	* If experiment on the model transfer ability from seen classes to unseen classes, the two classes maps can be downloaded from [class_map_seen.json](https://drive.google.com/file/d/19q9NnnCieycgfsLI-iQCdTu82oDRZTAO/view?usp=sharing) and [class_map_unseen.json](https://drive.google.com/file/d/15PEcgP15PC-1m6DAmEwiFnTDzGovzoRD/view?usp=sharing).

	* To train the model on Deep Fashion In-store dataset, run

	```
	python train_deepfashion.py --data ./deepfashion/ 
	--train-listfile ./train_listfile.json 
	--val-listfile ./val_listfile.json 
	--test-listfile ./test_listfile.json 
	--class-map-file ./classmap.json 
	--num-classes 17 
	--learning_rate 0.5 --temp 0.1
	--ckpt /pretrained_model/
	--dist-url 'tcp://localhost:10001' 
	--multiprocessing-distributed 
	--world-size 1 --rank 0 --cosine

	```

	* To evaluate the model, run
	```
	python eval_deepfashion.py --data ./deepfashion/ 
	--train-listfile ./train_listfile.json --val-listfile ./val_listfile.json 
	--class-map-file ./classmap.json 
	--num-classes 17 
	--learning_rate 0.5 --temp 0.1
	--ckpt /trained_model/

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