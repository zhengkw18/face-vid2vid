# face-vid2vid

## Usage

### Dataset Preparation

```shell
cd datasets
wget https://yt-dl.org/downloads/latest/youtube-dl -O youtube-dl
chmod a+rx youtube-dl
python load_videos.py --workers=8
cd ..
```

### Pretrained Headpose Estimator

[300W-LP, alpha 1, robust to image quality](https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR)

Put `hopenet_robust_alpha1.pkl` here

### Train

```shell
python train.py --batch_size=4 --gpu_ids=0,1,2,3 --num_epochs=100 (--ckp=10)
```

On 2080Ti, setting batch_size=4 makes up gpu memory

### Evaluate

Reconstruction：

```shell
python evaluate.py --ckp=99 --source=r --driving=datasets/vox/test/id10280#NXjT3732Ekg#001093#001192.mp4
```

The first frame is used as source by default

Motion transfer：

```shell
python evaluate.py --ckp=99 --source=test.png --driving=datasets/vox/test/id10280#NXjT3732Ekg#001093#001192.mp4
```

Face Frontalization：

```shell
python evaluate.py --ckp=99 --source=f --driving=datasets/vox/train/id10192#S5yV10aCP7A#003200#003334.mp4
```

## Acknowlegement
Thanks to [NV](https://github.com/NVlabs/face-vid2vid), [Imaginaire](https://github.com/NVlabs/imaginaire), [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model) and [DeepHeadPose](https://github.com/DriverDistraction/DeepHeadPose)
