from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

coco = COCO('all_info/coco_labels/21532_new.json')
#img_dir = '../datasets/coco/train2017'
image_id = 13766

img = coco.imgs[image_id]
#image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
#image = np.array(Image.open("data/copertoni/21532/train/good/2D.FE.DR.1.A_140341_GAMMA.obj.png"))
#plt.imshow(image, interpolation='nearest')
#plt.show()


#plt.imshow(image)
cat_ids = coco.getCatIds()
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids)
#coco.showAnns(anns) #se non commentato, crea problemi se vuoi mostrare solo la maschera
#plt.show()
#plt.clf()

mask = coco.annToMask(anns[0]) #initialize mask to all pixels took
for i in range(len(anns)): #for each type of tag
    mask += coco.annToMask(anns[i])

plt.imshow(mask)
plt.show()

plt.clf()
image_example = Image.open('data/copertoni/tire_split/masks/anomaly/20410_085504.obj_2D.FE.DR.1.B.png')
plt.imshow(image_example)
plt.show()