import numpy as np
import xml.etree.ElementTree as ET
import cv2
import skimage.io as io
import skimage.transform
import random
import os,sys
from processing.visualization import draw_ann 

myroot=os.getcwd()

sys.path.append(myroot)

class SVT:
	def __init__(self, trainPath=None, testPath=None):
		trainList = None
		testList = None
		if trainPath:
			self.trainList = self.parseTree(trainPath)
		if testPath:
			self.testList = self.parseTree(testPath)

	def parseTree(self, path):
		dataset = []
		tree = ET.parse(path)
		root = tree.getroot()

		for image in root.findall('image'):
			#存储的图片地址
			name = image.find('imageName').text
			rectangles = []
			tags=[]
			taggedRectangles = image.find('taggedRectangles')
			for rectangle in taggedRectangles.findall('taggedRectangle'):
				h = float(rectangle.get('height'))# / 300.0
				w = float(rectangle.get('width')) #/ 300.0
				x = float(rectangle.get('x')) #/ 300.0
				y = float(rectangle.get('y')) #/ 300.0
				tag=rectangle.find('tag').text
				rectangles.append(([x,y,w,h],0,tag))
				#tags.append(tag)
			dataset.append((name, rectangles))
		return dataset

	def create_batches(self,batch_size,dataset='train',shuffle=True):
        # 1 batch = [(image, [([x, y, w, h], id), ([x, y, w, h], id), ...]), ...]
		if dataset == 'train':
			datalist = self.trainList
		else:
			datalist = self.testList
		batch = []

		while True:
		    indices = range(len(datalist))
		    #print('len=',len(datalist))
		    if shuffle:
		        indices = np.random.permutation(indices)

		    for index in indices:
		        img = datalist[index][0]
		        path ='./svt1/'+img 
		        #print('img=',img)
		        I = io.imread(path)#
		        batch.append((I, datalist[index][1]))
		        if len(batch) >= batch_size:
		            yield batch #
		            batch = []
	def preprocess_batch(self, batch, augment=True):
		imgs = []
		all_used_anns = []
		#all_used_tags=[]
		image_size=300

		for img, anns in batch:
			used_anns = []
			#used_tags=[]
			w = img.shape[1]
			h = img.shape[0]

			option = np.random.randint(2)

			if not augment:
				option = 0

			if option == 0:
				sample = img #原始图片	 
			#进行压缩
			resized_img = skimage.transform.resize(sample, (image_size, image_size),mode='reflect')
			#并没有按照压缩的做
			for box,id,tag in anns:
				scaleX = 1.0 / float(sample.shape[1])
				scaleY = 1.0 / float(sample.shape[0])
				#转换成0-1之间的数字
				box0=box[0]*scaleX
				box1=box[1]*scaleY
				box2=box[2]*scaleX
				box3=box[3]*scaleY
				cX = box0 + box2 / 2.0
				cY = box1 + box3 / 2.0
				if cX >= 0 and cX <= 1 and cY >= 0 and cY <= 1:
					#used_anns.append((box,id,tag))
					used_anns.append(([box0,box1,box2,box3],id,tag))
					#used_tags.append(tag)
					#print('box={},{},{},{}'.format(box0,box1,box2,box3))
					#print('---------------------')
			imgs.append(resized_img)
			all_used_anns.append(used_anns)
			#all_used_tags.append(used_tags)
		return np.asarray(imgs), all_used_anns#,all_used_tags
	

if __name__ == '__main__':
	loader = SVT('./dataset/train.xml', './dataset/test.xml')

	batch = loader.create_batches(1,dataset='test',shuffle=False)

	for b in batch:
	    # [(image, [([x, y, w, h], id), ([x, y, w, h], id), ...]), ...]

		imgs, anns = loader.preprocess_batch(b,augment=False)

		I = imgs[0] * 255.0 

		for box_coords,id,tag in anns[0]:
			#print(tag)
			draw_ann(I, box_coords, tag)

		I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)
		#cv2.imshow("original image", cv2.cvtColor(b[0][0], cv2.COLOR_RGB2BGR))
		cv2.imshow("patch", I)
		if cv2.waitKey(0) == 27:
			break

