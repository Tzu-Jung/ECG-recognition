# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + id="PpSYpPh8cuqG" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 124} executionInfo={"status": "ok", "timestamp": 1591796282810, "user_tz": -480, "elapsed": 36743, "user": {"displayName": "\u9673\u54a8\u84c9", "photoUrl": "", "userId": "01605339006732320408"}} outputId="e93f0453-1c35-4582-b5cb-603ef7293b8d"
from google.colab import drive
drive.mount('/content/drive')

# + id="Q4LZ1USpbA3e" colab_type="code" colab={}
import pandas as pd
import os
import shutil  #用於移動檔案


#開啟表格檔案並讀取
f=open("/content/drive/My Drive/AI_Mango/C1-P1_Train Dev_fixed/C1-P1_Train/train.csv","rb")  #輸入表格所在路徑+名稱
list=pd.read_csv(f)
# list["FILE_ID_JPG"]=".jpg" #建立圖片名與類別相對應
#list["FILE_ID1"]=list["image_id"]

# list["FILE_ID1"]=list["image_id"]+list["FILE_ID_JPG"] #建立圖片名與類別相對應
#建立資料夾
'''
  for i in ['A', 'B', 'C']:
      os.mkdir(str(i))
'''
#進行分類
for i in ['A', 'B', 'C']:
    listnew=list[list["label"]==i]
    l=listnew["image_id"].tolist()
    j = os.path.join("/content/drive/My Drive/AI_Mango", i)
    os.makedirs(j, exist_ok=False)
    for each in l:
        shutil.copy(os.path.join("/content/drive/My Drive/AI_Mango/C1-P1_Train Dev_fixed/C1-P1_Train", each),j)




