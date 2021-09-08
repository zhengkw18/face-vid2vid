# face-vid2vid

### 数据集准备

```shell
cd datasets
wget https://yt-dl.org/downloads/latest/youtube-dl -O youtube-dl
chmod a+rx youtube-dl
python load_videos.py --workers=8
cd ..
```

数据源为Youtube，需要挂梯子

### 预训练姿态估计模型

[300W-LP, alpha 1, robust to image quality](https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR)

下载后将`hopenet_robust_alpha1.pkl`置于本目录下

### 训练

```shell
python train.py --batch_size=4 --gpu_ids=0,1,2,3 --num_epochs=100 (--ckp=10)
```

指定大于0的ckp时，将从指定epoch的Checkpoint加载

2080Ti上，batch size设置为4恰好占满显存

### 测试

重建：

```shell
python evaluate.py --ckp=99 --source=r --driving=datasets/vox/test/id10280#NXjT3732Ekg#001093#001192.mp4
```

默认使用第一帧为source重建

动作迁移：

```shell
python evaluate.py --ckp=99 --source=test.png --driving=datasets/vox/test/id10280#NXjT3732Ekg#001093#001192.mp4
```

正脸化：

```shell
python evaluate.py --ckp=99 --source=f --driving=datasets/vox/train/id10192#S5yV10aCP7A#003200#003334.mp4
```

## Acknowlegement
Thanks to [NV](https://github.com/NVlabs/face-vid2vid), [Imaginaire](https://github.com/NVlabs/imaginaire), [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model) and [DeepHeadPose](https://github.com/DriverDistraction/DeepHeadPose)
