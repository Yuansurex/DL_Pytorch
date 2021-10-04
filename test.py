# coding: utf-8
# @Time: 2021/10/3 22:33
# @Author: yuansure 
# @Email: 374487332@qq.com
# @Function: 

print("*" * 50)
import os
for i in range(3,11):
    #os.makedirs("ch"+str(i))
    os.removedirs("ch"+str(i))
