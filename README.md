# MPBTTrack



## Tracking performance
### Results on Multi-Mice PartsTrack dataset

#### mice part and body tracking
![mice](https://github.com/godHhh/MPBTTrack/assets/mice_part_and_body_tracking.mp4)

#### mice part tracking
![mice](https://github.com/godHhh/MPBTTrack/assets/mice_part_tracking.mp4)

#### mice body tracking
![mice](https://github.com/godHhh/MPBTTrack/assets/mice_body_tracking.mp4)

## News

- [2025.05.20] We are currently organizing the code and will follow up with an official release. Please stay patient and wait for further updates.


## Installation
### Setup with Anaconda
**Step 1.** Create Conda environment and install pytorch.
```shell
conda create -n MPBTrack python=3.8
conda activate MPBTrack
```
**Step 2.** Install torch and matched torchvision from https://pytorch.org/get-started/locally
The code was tested using torch 1.9.1+cu113 and torchvision==0.10.1 

**Step 3.** Install MPBTrack.
```shell
git clone https://github.com/godHhh/MPBTrack.git
cd MPBTrack
pip3 install -r requirements.txt
python3 setup.py develop
```
**Step 4.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip3 install cython; 
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
## Model Zoo
We provide pretrained model weights for MPBTrack. 

| Name | Model                                                                                                |
|-----|  ---------------------------------------------------------------------------------------------------- |
|  Multi-Mice PartsTrack  |  [Google Drive](https://drive.google.com/drive/folders/1dmhvc8hbx?usp=sharing) |


## Training
Download the COCO-pretrained YOLOX weight [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0) and put it under *\<MPBTTrack_HOME\>/pretrained*.

* **Train PartsTrack dataset model**

    ```shell
    python3 tools/train.py -f exps/example/mot/yolox_x_mice.py -d 1 -b 4 --fp16 -o -c pretrained/yolox_x.pth.tar
    ```


## Evaluation

* **on PartsTrack Test set**
    ```shell
    python tools/track.py -f exps/example/mot/yolox_mice_MPBReID.py -c pretrained/mice.pth.tar -b 1 -d 1 --fp16 --fuse --test
    ```
