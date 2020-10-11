import numpy as np
import cv2
from sklearn.cluster import KMeans
import pandas as pd

# places.csv
# ../val_256/Places365_val_00000001.jpg
# palette_7_places.csv #7: number of colors
#
### User inputs
num_clusters = 6
img_name = 'Kimetsu_no_Yaiba_Mugen_Ressha_Hen_Poster.jpg'
# img_path = '../dataset/test/' + img_name
img_path = './' + img_name
print(img_path)

###

img = cv2.imread(img_path)[:, :, [2, 1, 0]]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#for cv2, blue channel and red channel switches
# print(img.shape)#375,266,3
size = img.shape[:2]
if size != (256,256):
    img=cv2.resize(img,(256,256),interpolation = cv2.INTER_AREA)
    size= img.shape[:2]
# print(img.shape)#256,256,3
vec_img = img.reshape(-1, 3)
model = KMeans(n_clusters=num_clusters, n_jobs=-1)
pred = model.fit_predict(vec_img)
pred_img = np.tile(pred.reshape(*size,1), (1,1,3))

center = model.cluster_centers_.reshape(-1)
print(np.floor(center))

# Reshape for an input
print('img_name = \'%s\';' % img_name, end=" ")
for k, i in enumerate(model.cluster_centers_):
    print('manual_color_%d = [' % k + str(i[0].astype('int')) +', '+ str(i[1].astype('int'))+  ', '+ str(i[2].astype('int')) + '];', end=" ")
cv2.imwrite('./demonSlayer2.jpg',img)