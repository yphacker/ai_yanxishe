#coding=utf-8
import os
import os.path #文件夹遍历函数
#获取目标文件夹的路径
filedir = './Annotations_txt'
#获取当前文件夹中的文件名称列表
filenames=os.listdir(filedir)
#打开当前目录下的result.txt文件，如果没有则创建
f=open('result1.txt','w')

#先遍历文件名
path = 'xxx'# 图片路径
fileList = os.listdir(path)


for i in range(len(fileList)-1):

    filepath = filedir+'/'+ fileList[i].split(".")[0] +".txt"


    #遍历单个文件，读取行数
    for line in open(filepath):

        if line.strip() != "":
            f.writelines(line)

        f.write('\n')
#关闭文件
f.close()