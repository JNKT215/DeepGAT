# 深層な Graph Attention Networks における適切な Attention の学習

This repository is the implementation of [深層な Graph Attention Networks における適切な Attention の学習](https://www.jstage.jst.go.jp/article/jsaikbs/129/0/129_49/_article/-char/ja/ ).

> 深層な Graph Attention Networks における適切な Attention の学習．  
加藤 潤，猪口明博．   
第129回 知識ベースシステム研究会．


## Requirements
```bash
torch
torch_geometric
hydra
mlflow
tqdm
```

## Guide to experimental replication

CS dataset with DeepGAT
```bash
#best (num_layer=2, att_type=DP)
python3 train_coauthor.py key=GAT_cs_tuned_YDP_supervised_updated GAT_cs_tuned_YDP_supervised_updated.num_layer=2
#best (num_layer=2, att_type=SD)
python3 train_coauthor.py key=GAT_cs_tuned_YSD_supervised_updated GAT_cs_tuned_YSD_supervised_updated.num_layer=2
#max (num_layer=15, att_type=DP)
python3 train_coauthor.py key=GAT_cs_tuned_YDP_supervised_updated GAT_cs_tuned_YDP_supervised_updated.num_layer=15
#max (num_layer=15, att_type=SD)
python3 train_coauthor.py key=GAT_cs_tuned_YSD_supervised_updated GAT_cs_tuned_YSD_supervised_updated.num_layer=15
```

Physics dataset with DeepGAT
```bash
#best (num_layer=3, att_type=DP)
python3 train_coauthor.py key=GAT_physics_tuned_YDP_supervised_updated GAT_physics_tuned_YDP_supervised_updated.num_layer=3
#best (num_layer=2, att_type=SD)
python3 train_coauthor.py key=GAT_physics_tuned_YSD_supervised_updated GAT_physics_tuned_YSD_supervised_updated.num_layer=2
#max (num_layer=15, att_type=DP)
python3 train_coauthor.py key=GAT_physics_tuned_YDP_supervised_updated GAT_physics_tuned_YDP_supervised_updated.num_layer=15
#max (num_layer=15, att_type=SD)
python3 train_coauthor.py key=GAT_physics_tuned_YSD_supervised_updated GAT_physics_tuned_YSD_supervised_updated.num_layer=15
```

Flickr dataset with DeepGAT
```bash
#best (num_layer=4, att_type=DP)
python3 train_flickr.py key=GAT_Flickr_tuned_YDP_supervised_updated GAT_Flickr_tuned_YDP_supervised_updated.num_layer=4
#best (num_layer=4, att_type=SD)
python3 train_coauthor.py key=GAT_Flickr_tuned_YSD_supervised_updated GAT_Flickr_tuned_YSD_supervised_updated.num_layer=4
#max (num_layer=9, att_type=DP)
python3 train_flickr.py key=GAT_Flickr_tuned_YDP_supervised_updated GAT_Flickr_tuned_YDP_supervised_updated.num_layer=9
#max (num_layer=9, att_type=SD)
python3 train_coauthor.py key=GAT_Flickr_tuned_YSD_supervised_updated GAT_Flickr_tuned_YSD_supervised_updated.num_layer=9
```

PPI dataset with DeepGAT
```bash
#best,max (num_layer=9, att_type=DP)
python3 train_ppi.py key=GAT_ppi_tuned_YDP_supervised_updated GAT_ppi_tuned_YDP_supervised_updated.num_layer=9
#best,max (num_layer=9, att_type=SD)
python3 train_ppi.py key=GAT_ppi_tuned_YSD_supervised_updated GAT_ppi_tuned_YSD_supervised_updated.num_layer=9
```
If you need to know the parameters in detail, please check conf/config.yaml.