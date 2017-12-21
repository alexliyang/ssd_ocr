# ssd_ocr

Implement a fast text detector by tensorflow [paper](https://arxiv.org/pdf/1611.06779.pdf)

## Basic Instructions

1.Dataset directory contains 350 images,The annotations on all train and test images will be stored in two single xml file,all bounding boxes are represented by four parameters (x,y,w,h).
<br/>
2.You need download vgg16.npy from [this repository](https://github.com/machrisaa/tensorflow-vgg),which is better than you don't use.<br/>
3.some result are good and some bad ,Making this result maybe lack of amount of dataset,so keep much data as much as you can.<br/>
## Dependencies

* TensorFlow
* OpenCV
* XML

## Results of some test images
<img src="https://github.com/zhangcheng007/ssd_ocr/blob/master/images/00_23.jpg" width="300"/>
<img src="https://github.com/zhangcheng007/ssd_ocr/blob/master/images/00_25.jpg" width="300"/>
<img src="https://github.com/zhangcheng007/ssd_ocr/blob/master/images/00_30.jpg" width="300"/>


## References
https://github.com/seann999/ssd_tensorflow<br/>
https://github.com/shinjayne/textboxes
<br/>
