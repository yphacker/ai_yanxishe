
## score
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert(bert-base-uncased)|-1|-1|-1|

单折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|bert(bert-base-uncased),长文本|87.91|87.37|epoch=2就不再提升了|


## script
nohup python main.py -m='bert_gru' -b=4 -e=16 -mode=2 > nohup/bert_gru.out 2>&1 &  

python main.py -o=predict -m=bert_gru -b=32