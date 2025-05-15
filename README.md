# Step 1: Word-As-Image Environment

Follow [Word-As-Image](https://github.com/Shiriluz/Word-As-Image), prepare the environment

- reference: [【Word-As-Image】Stable Diffusionでフォントに意味を埋め込む](https://qiita.com/Yasu81126297/items/91edd41fcd2fb941743d)

```shell
git clone https://github.com/WordAsImage/Word-As-Image.git
cd Word-As-Image

conda create --name word python=3.8.15
conda activate word
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

pip install numpy scikit-image
pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom freetype-py shapely

pip install opencv-python==4.5.4.60
pip install kornia==0.6.8
pip install wandb
pip install shapely

pip install diffusers==0.8
pip install transformers scipy ftfy accelerate

git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive

# remove the last 8 lines of CMakeList.txt

conda install cmake
python setup.py install
```

# Step 2: Word-As-Image example

```shell
python code/main.py  --semantic_concept "BUNNY" --optimized_letter "Y" --font "KaushanScript-Regular" --seed 0
```

# Step 3: Prepare train and val dataset for discriminator

80% (3462+3462) train and 20% (866+866) val

- [process_dataset/prepare_train_and_val.ipynb](process_dataset/prepare_train_and_val.ipynb)

# Step 4: Dataloader for training discriminator

- [discriminator/dataset/dataloader.py](discriminator/dataset/dataloader.py)

# Stpe 5: Train discriminator

```shell
cd discriminator/
sh ./train.sh
```

# Step 6: Change SDSLoss to Discriminator loss

- [Word-As-Image/code/losses.py](Word-As-Image/code/losses.py)

# Step 7: Trans

- copy font file to `Word-As-Image/code/data/fonts/`

run

```shell
python code/main_app.py --semantic_concept "kuzushiji" --word "くずし字" --optimized_letter "字" --font "NotoSansJP-VariableFont_wght" --seed 0
```