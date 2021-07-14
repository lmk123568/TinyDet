<img src="demo/resouces/logo.png" alt="img1"  />

---

## 简介

**TinyDet**是一个简单的目标检测框架，数据集以VOC 2007为主，以实现魔改先进算法为主，快速验证idea有效性，以便在后续COCO进行验证，同时也有一些代码的注释，方便同学们后续学习

## 模型库

![model_zoo](demo\resouces\model_zoo.png)

## 已支持

- [ ] `ShuffleNet V2`
- [ ] `MobileNet V3`
- [ ] `ReXNet`
- [ ] `RepVGG`
- [ ] `YOLOV5_Backbone`
- [ ] `Coordinate Attention`
- [ ] `ACON`
- [x] `YOLOF`



## 安装

1. 照着官网完整安装 [MMdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)
2. 新建文件夹，Clone本仓库



## 数据集

本项目采用VOC 2007数据集，不包含VOC 2012

VOC07数据结构如下

```shell
└─VOCdevkit
    └─VOC2007
        ├─Annotations           
        ├─ImageSets             
        │  ├─Layout             
        │  ├─Main                
        │  │ ├─trainval.txt     # 用于train, imgs = 5011
        │  │ ├─test.txt         # 用于val,   imgs = 4952
        │  │  ...
        │  └─Segmentation        
        ├─JPEGImages            # total imgs = 9963
        ├─SegmentationClass      
        └─SegmentationObject     
```

为了方便同学下载数据，本项目新增了VOC数据自动下载脚本，没有VOC数据的同学可以运行

```shell
bash data/download_voc07.sh
```

将自动下载VOC07数据，无需任何后续操作



## 训练与测试

- 单卡训练

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py ${config}
```

- 多卡训练

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ${config} 4
```

- 测试

```shell
CUDA_VISIBLE_DEVICES=0 python tools/test.py ${config} ${checkpoint} --eval mAP
```

训练`(train, val)`用的是`configs/__base__/datasets/voc0712.py`中`data = dict(train=dict(), val=dict())`

测试(`test`)用的是`configs/__base__/datasets/voc0712.py`中`data = dict(test=dict())`

- 恢复训练

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py ${config} [--resume-from ${checkpoint}]
```

`--resume-form`同时加载模型权重和优化器状态，通常用于恢复意外中断的训练过程，若完整训练完后还要继续训练，可增大`max_epochs`，降低一下lr，重新开启训练

- Infer speed

推理速度包括`forwarding`和`postprocessing`，不包括`data loading time`，以`fps(img/s)`为单位

```shell
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py ${config} ${checkpoint}
```

`benchmark.py`将测试2000张图片下平均`fps`(默认开启`--fuse-conv-bn`)

- Demo

```shell
# img demo
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}]

# video demo
python demo/video_demo.py ${VIDEO_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--out ${OUT_FILE}]

# webcam demo
python demo/webcam_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--camera-id ${CAMERA-ID}]
```

## 工具

- 日志分析

```shell
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]

# example
python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.jpg
```

- 模型复杂度`FLOPs`

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

FLOP与输入形状有关，而参数与输入形状无关。默认输入形状为`(1, 3, 1280, 800)`


## FAQ

- 训练时候`Batch size`尽可能大，以使得`BN`有效

训练时候`loss`出现Nan，要么是数据集问题，要么是学习率问题，`MMdetection` 默认8卡训练，如果单卡或双卡需要改变学习率

| `gpu` | `samples_per_gpu` | `batch size` | `lr`   | `max iter` | `warm up` |
| ----- | ----------------- | ------------ | ------ | ---------- | --------- |
| 1     | 2                 | 2            | 0.0125 | 180000*8   | 4000      |
| 2     | 2                 | 4            | 0.025  | 180000*4   | 2000      |
| 4     | 2                 | 8            | 0.01   | 180000*2   | 1000      |
| 8     | 2                 | 16           | 0.02   | 180000     | 500       |

- 突然中断训练，但服务器进程还在跑，可以运行

```shell
kill -9 ${进程号}
```

- `dataset`部分配置尽可能在`__base__`修改，不要在主文件修改，否则会报错

```shell
[Errno 2] No such file or directory: '/tmp/tmpscs77a8_/tmp8vgjpul4.py'
```

- 动态观察`nvidia-smi`

```shell
watch -n 1 nvidia-smi
```

- 多卡训练报错`RuntimeError: Address already in use`

```shell
PORT=29501 # 改变端口即可，通常+1
```

- 修改VOC训练需要注意的点

```python
# 类别数要调整
# 预训练模型要加上，这个影响很大，有时候模型难以收敛
# 数据集路径要对齐
# lr 和 warmup 要调整
# 注意cfg文件中两个训练方式
```

