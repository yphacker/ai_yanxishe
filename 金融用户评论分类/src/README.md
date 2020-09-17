## score
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert(bert-base-uncased)|0.8839|88.60|0.8788,0.8933,0.8823,0.8817,0.8835|

单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert(bert-base-uncased)|88.62|88.19|epoch=4就不再提升了|
|longformer+512|85.43|85.19|epoch=10就不再提升了|



## script
max_seq_len = 256
nohup python main.py -m='bert' -b=25 -e=4 -mode=2 > nohup/bert.out 2>&1 &  
python main.py -o=predict -m=bert -b=32  

nohup python main.py -m='bert' -b=25 -e=4 > nohup/bert.out 2>&1 &  
python predict.py -m=bert
nohup python main.py -m='bert' -b=8 -e=4 > nohup/bert.out 2>&1 & 

max_seq_len = 512
nohup python main.py -m='longformer' -b=100 -e=10 -mode=2 > nohup/longformer.out 2>&1 &  
python main.py -o=predict -m=longformer -b=32 

max_seq_len = 1024
nohup python main.py -m='longformer' -b=50 -e=10 -mode=2 > nohup/longformer.out 2>&1 &  
python main.py -o=predict -m=longformer -b=32 