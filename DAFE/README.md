# DAFE

```python
# Training on Occluded-Duke
python train.py --config_file configs/OCC_Duke/vit_transreid_stride.yml 

# Training on Market-1501
python train.py --config_file configs/Market/vit_transreid_stride.yml 

# Testing on Occluded-Duke
python test.py --config_file configs/OCC_Duke/vit_transreid_stride.yml  TEST.WEIGHT 'log/lr0008_b32/transformer_best.pth'

```


