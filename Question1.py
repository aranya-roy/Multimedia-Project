from PIL import Image
import numpy as np

#Grey Scale
img = np.array(Image.open('Tiger.jpg').convert('RGB'))
gray = ((img[:,:,0] + img[:,:,1] + img[:,:,2]) // 3).astype('uint8')
Image.fromarray(gray).save('gray_scale.png')

#Median Cut
img = Image.open('Tiger.jpg')
q = img.quantize(colors=16, method=Image.MEDIANCUT)
q.save('median_cut.png')

#Octree
Image.open('Tiger.jpg').quantize(colors=16, method=Image.FASTOCTREE).save('octree.png')