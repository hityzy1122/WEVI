## Training Weakly Supervised Video Frame Interpolation with Events
(accepted by ICCV2021)

[[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_Training_Weakly_Supervised_Video_Frame_Interpolation_With_Events_ICCV_2021_paper.html)]
[[Video](https://www.youtube.com/watch?v=ktG5U3WKGes&t=2s)]

### 1.Abstract
This version of code is used for training on real low-fps data of dvs, which is collected by [DAVIS240C](https://inivation.com/wp-content/uploads/2019/08/DAVIS240.pdf). This code can be trained by the visible low-fps frames(12fps) with corresponding events saved in the aedat4 files  and interpolate the inbetweens at any time. An aedat4 file is provided in dataset/aedat4, which can be used as a demo to run the whole process.
 
Sorry for breaking the promise. As some sensitive information about face, human body, number plate and palm print exists in most of the proposed slomoDVS dataset, the dataset and the pretained weight on it do not pass the compliance review policy launched recently by company.
### 2.Environments
1) cuda 9.0

2) python 3.7

3) pytorch 1.1

4) numpy 1.17.2

5) tqdm

6) gcc 5.2.0

7) cmake 3.16.0

8) opencv_contrib_python

9) compiling correlation module
(The PWCNet and the correlation module are modified from [DAIN](https://github.com/baowenbo/DAIN/tree/master/PWCNet))

   a) cd stage1/lib/pwcNet/correlation_pytorch1_1

   b) python setup.py install


10) Install apex: https://github.com/NVIDIA/apex

11) For processing DVS file:
   
     a) More detail information about aedat4 file and DAVIS240C can be found in [here](https://inivation.gitlab.io/dv/dv-docs/docs/getting-started/)

     b) tools for processing aedat4 file: [dv-python](https://gitlab.com/inivation/dv/dv-python)

12) For distributed training with multi-gpus on cluster: slurm 15.08.11
 
### 3.Preparing training data
You can prepare your own event data according to the demo in DVSTool

1) Place aedat4 file in ./dataset/aedat4
2) cd DVSTool
3) python mainDVSProcess_01.py  
It will extract the events and frame saved in .aedat4 into pkl which will be saved in dataset/fastDVS_process
4) python mainGetDVSTrain_02.py  
It will gather the train samples and save in dataset/fastDVS_dataset/train.  (A train sample includes I0, I1, I2, I01, I21 and E1)
5) python mainGetDVSTest_03.py  
It will gather the test samples and save in dataset/fastDVS_dataset/test  (A test sample includes I_-1, I0, I1, I2, E1/3, E2/3)
### 4.Training stage1
cd stage1 
#### 1) Training with single gpu:
a) Modify the config in configs/configEVI.py accordingly

b) python train.py

#### 2) Training with muli-gpus(16) on cluser managed by slurm:
a) Modify config in configs/configEVI.py accordingly

b) Modify runEvi.py in runBash accordingly

c) python runBash/runEvi.py

### 5.Training stage2
cd stage2 

Place the experiment dir trained by stage1 in ./output

#### 1) Training with single gpu:
a) Modify the config in configs/configEVI.py accordingly, especially the path in line 28, 29

b) python train.py

#### 2) Training with muli-gpus(16) on cluser managed by slurm:
a) Modify config in configs/configEVI.py accordingly, especially the path in line 28, 29

b) Modify runEvi.py in runBash accordingly

c) python runBash/runEvi.py

### 6. Citation 
```
@InProceedings{Yu_2021_ICCV,
    author    = {Yu, Zhiyang and Zhang, Yu and Liu, Deyuan and Zou, Dongqing and Chen, Xijun and Liu, Yebin and Ren, Jimmy S.},
    title     = {Training Weakly Supervised Video Frame Interpolation With Events},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14589-14598}
}
```
### 7. Reference code base 
[[styleGAN](https://github.com/tomguluson92/StyleGAN_PyTorch)], [[TTSR](https://github.com/researchmm/TTSR)], [[DAIN](https://github.com/baowenbo/DAIN/tree/master/PWCNet)], [[superSlomo](https://github.com/avinashpaliwal/Super-SloMo)], [[QVI](https://sites.google.com/view/xiangyuxu/qvi_nips19)], [[faceShiter](https://github.com/mindslab-ai/faceshifter)]
