import pandas as pd
import cv2
from PIL import Image,ImageDraw
train = pd.read_csv("../data/train_bboxes.csv")
print(train.head(10))
res = pd.DataFrame()
ID = []
bbox = []

for i in range(20013):
    dealres = train[train['filename'] == i]

    list = ""
    for j in range(dealres.shape[0]):
        list = list + str(dealres['x'].iloc[j]) + " " + str(dealres['y'].iloc[j]) + " "+ str(dealres['x'].iloc[j]+dealres['width'].iloc[j]) + " "+ str(dealres['y'].iloc[j]+dealres['height'].iloc[j]) \
               + " " + str(1.0)+ ";"
    if dealres.shape[0]>0:
        ID.append("/media/hszc/data/datafountain/lungs/data/train/"+str(i)+".jpg")
        bbox.append(list[:-1])
    else:
        ID.append("/media/hszc/data/datafountain/lungs/data/train/" + str(i) + ".jpg")
        bbox.append("0.0 0.0 10.0 10.0 2.0")
res['ID'] = ID
res['bbox'] = bbox
res.to_csv("train.csv",index=False)

