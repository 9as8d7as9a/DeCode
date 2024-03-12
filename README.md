# DeCode
This is the official code for "Let Me DeCode You: Decoder Conditioning with Tabular Data".

## Overview
![Figure 1. Method overview](figures/DeCode_overview.png?raw=true "DeCodeOverview")

## Abstract
Training deep neural networks for 3D segmentation tasks can be challenging, often requiring efficient and effective strategies to improve model performance. In this study, we introduce a novel approach, DeCode, that utilizes label-derived features for model conditioning to support the decoder in the reconstruction process dynamically, aiming to enhance the efficiency of the training process. DeCode focuses on improving 3D segmentation performance through the incorporation of conditioning embedding with learned numerical representation of 3D-label shape features. Specifically, we develop an approach, where conditioning is applied during the training phase to guide the network toward robust segmentation. When labels are not available during inference, our model infers the necessary conditioning embedding directly from the input data, thanks to a feed-forward network learned during the training phase. This approach is tested using synthetic data and cone-beam computed tomography (CBCT) images of teeth. For CBCT, three datasets are used: one publicly available and two in-house. Our results show that DeCode significantly outperforms traditional, unconditioned models in terms of generalization to unseen data, achieving higher accuracy at a reduced computational cost. This work represents the first of its kind to explore conditioning strategies in 3D data segmentation, offering a novel and more efficient method for leveraging annotated data. We provide the community with access to both the source code and pre-trained models, encouraging further exploration and development in this promising area.

## 3DeCode Synthethic Dataset
![3DeCode dataset examples](figures/3decode.png?raw=true "3DeCode")


## Reproducibility
1. To install dependencies:
```
 conda env create -f environment_decode.yml 
```
2. Training CBCT dataset source: https://www.nature.com/articles/s41467-022-29637-2
3. Dataset split IDs: ```config/data_split.json```
4. 3DeCode dataset - to generate syntethic dataset for all conditioning tasks run:
```
python src/data_utils/3decode_dataset.py
```
5. All necessary variables are stored in configuration files.
```
config/general_config.yaml
config/3decode_config.yaml
```
6. To calculate shape features from labels run:
```python src/data_utils/shape_radiomics.py```
7. To train CBCT segmentation model with default parameters run:
```python src/train.py```
8. To train 3DeCode synthetic data conditioning experiment run:
```python src/train_3decode.py```
9. To recreate all results use configured training batches, to use provided shell scripts first make them executable eg.: ```chmod +x ./experiments_cuda0_cbct_table2.sh```