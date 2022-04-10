# DSSP

This repository provides the code for the paper *Hyperspectral Image Reconstruction Using a Deep Spatial-Spectral Prior.* (CVPR 2019).[[Link]](https://ieeexplore.ieee.org/document/8954038)

## Environment

Python 3.7.9<br/>
CUDA 10.1<br/>
TensorFlow 1.14.0<br/>
h5py 2.10.0<br/>

## Train

### Dataset Preparation

To train the DNU model for hyperspectral imaging, the datasets should be downloaded to your computer in advance.
(e.g., [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/), [KAIST](http://vclab.kaist.ac.kr/siggraphasia2017p1/), [ICVL](http://icvl.cs.bgu.ac.il/hyperspectral/), and [Harvard](http://vision.seas.harvard.edu/hyperspec/index.html).)

For  ICVL dataset,  you can randomly select 100 spectral images for training and 50 spectral images for testing.  For  Harvard dataset, you should remove 6 deteriorated spectral images with large-area saturated pixels, and randomly select 35 spectral images for training and 9 spectral images for testing.
The training and test images in the ICVL dataset and  Harvard dataset are 48 * 48 inclined image blocks. 

In the  newly increased experiments, the CAVE dataset is used for training and the KAIST dataset is used for test. Then, you should modify the original CAVE and KAIST datasets by spectral interpolation, which have 28 spectral bands ranging from 450nm to 650nm. The patch size for training is 48*48, and the patch size for test is 256*256.

Then, edit the ```Training_data_Name``` in **train.py** to indicate the name and path to your dataset. Here is an example:
```
Training_data_Name = "/PATH/DATSET", 
```
And there should be two directories in your dataset path: [train, test] to indicate which part should be used for training and testing.


### Start Training

The training can be started with the following commands:
```bash
# For Harvard and ICVL dataset, using the trian set
python Train.py 

# For CAVE and KAIST dataset, using the train set
python Train_cave.py 
```

When the training starts, the trainer will save checkpoints into **./ckpt/** 

The checkpoint of three DSSP models trained on Harvard and CAVE is provided in **./pretrained_model/**. You can directly apply them to subsequent test.

## Test

After training, reconstruction image can be generated using the following commands:
```bash
# For Harvard and ICVL dataset, using the test set
python Test.py 

# For CAVE and KAIST dataset, using the test set
python Test_kaist.py 
```

## Results
### 1. Reproducing Results on Harvard and ICVL Dataset
The results of paper on [Harvard Dataset](http://vision.seas.harvard.edu/hyperspec/) and [ICVL Dataset](http://icvl.cs.bgu.ac.il/hyperspectral/). In this stage, the mask is randomly generated for each batch. And the size of patches is 48 * 48 * 31. In addition, only the central areas with 512* 512 * 31 are compared in testing.
<table align="center">
   <tr align = "center">
      <td></td>
      <td>Harvard</td>
      <td>ICVL</td>
   </tr>
   <tr align = "center">
      <td>PSNR</td>
      <td>32.84</td>
      <td>34.13</td>
   </tr>
   <tr align = "center">
      <td>SSIM</td>
      <td>0.979</td>
      <td>0.992</td>
   </tr>
   <tr align = "center">
      <td>SAM</td>
      <td>0.089</td>
      <td>0.028</td>
   </tr>
</table>

### 2. Results of Extra-Experiments on CAVE&KAIST Datasets
For academic reference, we have added some comparisons with the latest methods on [CAVE Dataset](https://www1.cs.columbia.edu/CAVE/projects/gap_camera/) and [KAIST Dataset](http://vclab.kaist.ac.kr/siggraphasia2017p1/). Methods for comparison include [TSA](https://github.com/mengziyi64/TSA-Net/) and [DGSM](https://github.com/TaoHuang95/DGSMP), and  our method is completely consistent with the experimental setup of these methods. In addition, we have also increased the comparison of using different masks. In "Real-mask", a given real mask in the range of 0-1 is utilized, which is provided by [TSA](https://github.com/mengziyi64/TSA-Net/tree/master/TSA_Net_realdata/Data). In "Binary-mask", the given real mask is rounded to a binary mask. When training the model, a 48 * 48 sub-mask should be randomly derived from the given real mask for each batch. Note that, images with a size of 256 * 256 * 28, matched the given real mask, are used for comparison.
<table align="center">
   <tr align = "center">
      <td  rowspan="2"></td>
      <td>TSA</td>
      <td>DGSM</td>
      <td colspan="2">DSSP </td>
   </tr>
   <tr align = "center">
      <td>Real-mask</td>
      <td>Real-mask</td>
      <td>Real-mask</td>
      <td>Binary-mask</td>
   </tr>
   <tr align = "center">
      <td>PSNR</td>
      <td>31.46</td>
      <td>32.63</td>	
      <td>32.39</td>
      <td>32.84</td>
   </tr>
   <tr align = "center">
      <td>SSIM</td>
      <td>0.894</td>
      <td>0.917</td>
      <td>0.971</td>
      <td>0.974</td>
   </tr>
   <tr align = "center">
      <td>SAM</td>
      <td>-</td>
      <td>-</td>
      <td>0.177</td>
      <td>0.163</td>
   </tr>
</table>

# Citation
If our code is useful in your reseach work, please consider citing our paper.
```
@inproceedings{DSSP,
  title={Hyperspectral Image Reconstruction Using a Deep Spatial-Spectral Prior},
  author={Lizhi Wang, Chen Sun, Ying Fu, Min H. Kim and Hua Huang, },
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8024-8033},
  year={2019}
}
```

