import shutil
from objectdatasetscript import get_mask_patch
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
from numpy import asarray

#src = "models/tmp/object_model"
#dst = "data/"

#copiare ultimo weight
src = "weights/object_model9999.pth"
dst = "data/"

#copiare report
#src = "./report.txt"
#dst = "data/"



#src = "study_elements/auprc.jpg"
#dst = "data/"

#copiare immagine mask
'''
src = "data_splitted/copertoni/object_split/masks/anomaly/21373_113836.obj_2D.FE.DR.1.B.png"
img = Image.open(src)
fig, subplots = plt.subplots(1, 1)
plt.imshow(img)
values = asarray(img)
values = set(values.flatten())
print(values)
dst = "data/"
plt.savefig(os.path.join(dst,"mask_greyscale.jpg"))
'''

#copiare immagine sidelight
#src = "sidelight/object_split/1/validation/anomaly/20410_125603.obj_P_1_2_2D.FE.DR.1.B_001.png"
#dst = "data/"

#copiare immagine da validation
#src = "data_splitted/copertoni/object_split/validation/anomaly/20410_063918.obj_P_2_3_2D.FE.DR.1.A.png"
#dst = "data/"

shutil.copy(src, dst)

