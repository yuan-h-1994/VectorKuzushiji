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
# Step 2: Download MXFont

Download MXFont follow https://github.com/clovaai/mxfont and put it into Word-As-Image/code/mxencoder

# Step 3: Prepare data

Change the character in the config. Prepare the content image using ttf2img.py before generation. Content images are stored in data/melody.

The Kuzushiji data samples are in the data folder, and the style image can be modified by modifying "target_img_path" in Word-As-Image/code/config/base.yaml.

# Step 4: Run
```shell
cd Word-As-Image
python code/main_Kuzushiji.py
```
