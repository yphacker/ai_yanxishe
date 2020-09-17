
## score
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert-base+bigru+labelsmoothing|0.9624|89.77|b=256;0.9626,0.9596,0.9623,0.9631,0.9645|
|bart+bigru+labelsmoothing|0.9459|91.09|b=256;dropout:0.8;0.9505,0.9445,0.9417,0.9458,0.9471|

bert-base单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bilstm|89.39|89.37|b=256|
|bigru|89.39|89.66|b=256|
|bigru(avg+max)|89.33|89.37|b=256|
|bilstm+labelsmoothing|89.39|89.57|b=256|
|bigru+labelsmoothing|89.50|89.72|b=256|
|bigru(avg+max)+labelsmoothing|89.50|89.26|b=256|
|bilstm+FocalLoss|89.39|89.37|b=256|

## script
nohup python main.py -m='bert' -b=32 -e=4 -mode=2 > nohup/bert.out 2>&1 &  
nohup python main.py -m='bert' -b=8 -e=2 -mode=2 > nohup/bert.out 2>&1 &  
python main.py -o=predict -m=bert -b=32
nohup python main.py -m='bart' -b=8 -e=4 -mode=2 > nohup/bart.out 2>&1 &  
nohup python main.py -o=predict -m=bart -b=2 > info.out 2>&1 &  

python main2.py -m='rnn' -b=256 -e=16 -mode=2
python main2.py -o=predict -m=rnn -b=32  
python main2.py -m='rnn' -b=256 -e=16  
python predict2.py -m=rnn  

python predict.py -type=files -files=submission_0.csv,submission_1.csv,submission_2.csv,submission_3.csv