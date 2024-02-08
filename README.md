# S2M : Human motion generation 


[//]: # (![teaser]&#40;https://github.com/GuyTevet/mdm-page/raw/main/static/figures/github.gif&#41;)

#### Our repository is established on the basis of [Motion-Diffusion-Model](https://github.com/GuyTevet/motion-diffusion-model).


## Getting started

### 1. set up environment
Setup conda env:
```shell
conda env create -f environment.yml
conda activate s2m
```

Download dependencies:


```shell
bash prepare/download_smpl_files.sh
```


### 2. Get data

<details>
  <summary><b>HumanML3D</b></summary>

There are two paths to get the data:

(a) **Go the easy way if** you just want to generate text-to-motion (excluding editing which does require motion capture data)

(b) **Get full data** to train and evaluate the model.


#### a. The easy way (text only)

**HumanML3D** - Clone HumanML3D, then copy the data dir to our repository:

```shell
cd ..
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D motion-diffusion-model/dataset/HumanML3D
cd HumanMotionGeneration
```


#### b. Full data (text + motion capture)

**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./test_data
```

**KIT** - Download from [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git) (no processing needed this time) and the place result in `./dataset/KIT-ML`
</details>

<details>
  <summary><b>Sketches</b></summary>

**generate sketches for HumanML3D dataset** 
```bash
python -m data_loaders.humanml.utils.plot_train
```
The sketches will be saved under `./test_data/sketches`
</details>

### 3. Get pre-trained model 
[Pre-trained model](https://drive.google.com/file/d/1nJbnQX5RLLLNsPvTX47PP6T1vko70t-9/view?usp=drive_link)
Put this pre-trained model under `./user_output`


## To be finised
