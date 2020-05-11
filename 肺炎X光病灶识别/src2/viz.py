import pandas as pd
import cv2
from PIL import Image,ImageDraw
train = pd.read_csv("../data/train_bboxes.csv")
print(train.head(10))

for i in (train['filename'].unique()):
    dealres = train[train['filename'] == i]
    img = cv2.imread("/media/hszc/data/datafountain/lungs/data/train/"+str(i)+".jpg")
    for j in range(dealres.shape[0]):
        xmin = int(dealres['x'].iloc[j])
        xmax = int(dealres['x'].iloc[j]+dealres['width'].iloc[j])
        ymin = int(dealres['y'].iloc[j])
        ymax = int(dealres['y'].iloc[j]+dealres['height'].iloc[j])
        img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,255,0),2)
    cv2.imwrite("/media/hszc/data/datafountain/lungs/data/label/"+str(i)+".jpg",img)

