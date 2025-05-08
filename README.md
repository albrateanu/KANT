# [Sensors 2025] Enhancing Low-Light Images with Kolmogorov–Arnold Networks in Transformer Attention

<div align="center">
  
[![MDPI](https://img.shields.io/badge/MDPI-paper-blue)](https://www.mdpi.com/1424-8220/25/2/327)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhancing-low-light-images-with-kolmogorov/low-light-image-enhancement-on-lolv2)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lolv2?p=enhancing-low-light-images-with-kolmogorov)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhancing-low-light-images-with-kolmogorov/low-light-image-enhancement-on-lolv2-1)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lolv2-1?p=enhancing-low-light-images-with-kolmogorov)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enhancing-low-light-images-with-kolmogorov/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=enhancing-low-light-images-with-kolmogorov)

</div>

**Note**: Entries of KANT on paperswithcode.com show the best results obtained through frequent validation. Paper shows average performance with less frequent validation. 


## 1. Create Environment

- Make Conda Environment
```
conda create -n KANT python=3.7
conda activate KANT
```

- Install Dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```


### 2. Prepare Datasets
Download the LOLv1 and LOLv2 datasets:

LOLv1 - [Google Drive](https://drive.google.com/file/d/1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu/view?usp=sharing)

LOLv2 - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

**Note:** Under the main directory, create a folder called ```data``` and place the dataset folders inside it.
<details>
  <summary>
  <b>Datasets should be organized as follows:</b>
  </summary>

  ```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |    ...
  ```

</details>

## 3. Testing

```
# LOL-v1
python3 Enhancement/test_from_dataset.py --opt Options/KANT_LOLv1.yml --weights pretrained_model/KANT_LOLv1.pth --dataset LOLv1 --self_ensemble --GT_Mean

# LOL-v2-real
python3 Enhancement/test_from_dataset.py --opt Options/KANT_LOLv2R.yml --weights pretrained_model/KANT_LOLv2R.pth --dataset LOLv2R --self_ensemble --GT_Mean

# LOL-v2-synthetic
python3 Enhancement/test_from_dataset.py --opt Options/KANT_LOLv2S.yml --weights pretrained_model/KANT_LOLv2S.pth --dataset LOLv2S --self_ensemble --GT_Mean
```

## 4. Training

```
# LOL-v1
python3 basicsr/train.py --opt Options/KANT_LOLv1.yml

# LOL-v2-real
python3 basicsr/train.py --opt Options/KANT_LOLv2R.yml

# LOL-v2-synthetic
python3 basicsr/train.py --opt Options/KANT_LOLv2S.yml

# Or, for distributed GPU training
bash train_multigpu.sh Options/KANT_[LOLv1|LOLv2R|LOLv2S].yml [GPU_id] [port, e.g. 4321]
# example:
bash train_multigpu.sh Options/KANT_LOL_v2S.yml 0 4321 # to train on LOL-v2-synthetic on GPU 0
```

## 5. Citation
```
@Article{brateanu2025kant,
  AUTHOR = {Brateanu, Alexandru and Balmez, Raul and Orhei, Ciprian and Ancuti, Cosmin and Ancuti, Codruta},
  TITLE = {Enhancing Low-Light Images with Kolmogorov–Arnold Networks in Transformer Attention},
  JOURNAL = {Sensors},
  VOLUME = {25},
  YEAR = {2025},
  NUMBER = {2},
  ARTICLE-NUMBER = {327},
  URL = {https://www.mdpi.com/1424-8220/25/2/327},
  ISSN = {1424-8220},
  DOI = {10.3390/s25020327}
}

@InProceedings{brateanu2024kant,
  author={Brateanu, Alexandru and Balmez, Raul},
  booktitle={2024 International Symposium on Electronics and Telecommunications (ISETC)}, 
  title={Kolmogorov-Arnold Networks in Transformer Attention for Low-Light Image Enhancement}, 
  year={2024},
  pages={1-4},
  keywords={Deep learning;Attention mechanisms;Transformers;Image restoration;Telecommunications;Image enhancement;Context modeling;Image restoration;Low-light enhancement;Vision transformer},
  doi={10.1109/ISETC63109.2024.10797300}}
```


```
