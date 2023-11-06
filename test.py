import sys
import numpy as np
import cv2 as cv
import pandas as pd
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import math

input_image = cv.imread("res_b.jpg")

image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)

rows,columns,channels = image.shape
print ('Image shape: ', image.shape)

mean_color = cv.mean(image)
print ('mean: ', mean_color)

# cv.imshow("Input Image", image)
# cv.waitKey(0

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
# print('model', model)

# train = pd.read_csv("annotations.csv")
# print(train.head())

num_dark_pixels = [0 for i in range(columns)]
sum_dark_pixels = [0 for i in range(columns)]
avg_dark_pixels = [0 for i in range(columns)]

global num
for x in range(0,columns):
  num = 0
  for y in range(0,rows):
    res = image[y,x]
    if res is None:
      continue
    if res[0] > mean_color[0] * 0.5 and res[1] > mean_color[1] * 0.5 and res[2] > mean_color[2] * 0.5:
      pass
    else:
      num += 1
      sum_dark_pixels[x] += y

  if num > 2:
    num_dark_pixels[x] = num
    avg_dark_pixels[x]=((sum_dark_pixels[x]) / num)

        
average_dark_pixels = 0
count = 0
for x in range(0, columns):
  if num_dark_pixels[x] > 0:
    average_dark_pixels += num_dark_pixels[x] 
    count += 1 

x_min = -1
x_max = 0

if count > 0:
    average_dark_pixels /= count 
    for x in range(0,columns):
        if num_dark_pixels[x] > average_dark_pixels + 15: 
            if x_min < 0:
                x_min = x 
            x_max = x 

if x_min >= 0: 
    y_min = avg_dark_pixels[x_min] 
    y_max = avg_dark_pixels[x_max]
    x_min_2 = x_min - 20

    if x_min_2 < 0:
        x_min_2 = 0
    if x_min_2 > len(avg_dark_pixels) - 1:
        x_min_2 = len(avg_dark_pixels) - 1

    x_max_2 = x_max + 20
    if x_max_2 < 0:
        x_max_2 = 0
    if x_max_2 > len(avg_dark_pixels) - 1:
        x_max_2 = len(avg_dark_pixels) - 1

    ratio = (avg_dark_pixels[x_max_2] - avg_dark_pixels[x_min_2]) / ((x_max_2) - (x_min_2))
    y_min = (int) (ratio * (x_min - x_min_2) + avg_dark_pixels[x_min_2])
    y_max = (int) (ratio * (x_max - x_max_2) + avg_dark_pixels[x_max_2])

    w = x_max - x_min
    
    if w > 10 and ratio < 0.7 and ratio > -0.7:
        x1=min(x_min,x_max)
        x2=max(x_min,x_max)
        y1=min(y_min+15,y_max+15,y_max-15,y_min-15)
        y2=max(y_min+15,y_max+15,y_max-15,y_min-15)

        
        res = cv.resize(image[y1:y2, x1:x2],(columns,50),fx=0, fy=0, interpolation = cv.INTER_NEAREST)#cut out the colour bands part from the image
        cv.rectangle(image, (x1,y1), (x2,y2),(0, 255, 0),2); #draw a green rectangle where the part of image was extracted on original image

        n = 3
        
        res_rows,res_columns,res_channels = res.shape
        background = [0 for k in range(n)]
        
        for x in range(res_columns-50,res_columns):
            value = res[0, x]
            for i in range(0,3):
                background[i] += value[i] / 50.0
        
        background_distances = [0 for k in range(res_columns)]
        
        average_distance = 0
        for x in range(0,res_columns-5):
            c1 = background
            c2 = res[0,x]
            
            background_distances[x] = (math.pow(c1[0] - c2[0], 2)
                          + math.pow(c1[1] - c2[1], 2)
                          + math.pow(c1[2] - c2[2], 2))/100
            average_distance += (background_distances[x] / res_columns)
         

        
        res= cv.fastNlMeansDenoising(res, None, 10, 7, 30)
        res = cv.GaussianBlur(res,(15,15),0)
       
        res = cv.GaussianBlur(res,(15,15),0)
        
        cv.imshow('res',cv.cvtColor(res, cv.COLOR_BGR2RGB))
       
        bright_codes = [(0, 0, 0),         #black
                        (156, 102, 51), #brown
                     (255, 0, 0),       #red
                     (255, 102, 0),     #orange
                     (255, 255, 0),     #yellow
                     (0, 255, 0),       #green
                     (0, 0, 255),       #blue
                     (200, 0, 255),     #violet
                     (128, 128, 128),   #gray
                     (255, 255, 255)]   #white

        dark_codes = [(20, 20, 20),      #black
                 (71, 53, 38),      #brown
                 (204, 0, 0),       #red
                      (255, 51, 0),     #orange
                    (255, 204, 102), #yellow
                      (30, 200, 50), #green
                 (40, 73, 86),      #blue
                 (110, 0, 51),      #violet
                 (73, 65, 62),      #gray
                 (250, 250, 250)]   #white

        color_name = ["black","brown","red","orange","yellow","green","blue","violet","gray","white"]
        #getting colors of the bands
        coldet = [0]*res_columns
        for x in range(0,res_columns):
            if background_distances[x] > average_distance :
                col = res[0, x]
                
                mini = 100000
                minc = -1
                for c in range(0,len(dark_codes)):
                    c2 = dark_codes[c]
                    distance = (math.pow(col[0] - c2[0], 2)
                            + math.pow(col[1] - c2[1], 2)
                            + math.pow(col[2] - c2[2], 2))
                    if distance < mini:
                        mini = distance
                        minc = c
                        
                coldet[x] = minc
                if minc >= 0:
                    pass
                   
            else:
                coldet[x] = -1

        print("coldet",coldet)

        numconti = 0
        numcodes = 0
        sumcodes = [0]*res_columns
        result = [0 for k in range(n)]
        found = False

        for x in range(0,res_columns):
            if coldet[x] == -1 and numconti > 20:
                sumc = [0]*len(dark_codes)
                for i in range(0,numconti - 20):
                    sumc[sumcodes[i]] += 1
                print("sumc",sumc)

                maxnum = 0
                code = -1
                for i in range(0,len(dark_codes)):
                    if sumc[i] > maxnum:
                        maxnum = sumc[i]
                        code = i

                if code is not -1:
                   #display colours and colour names
                    cv.rectangle(image, ((int)(numcodes*columns/4),rows-40), ((int)((numcodes +1 )*columns/4),rows),bright_codes[code],-1)#Original
                    cv.putText(image, color_name[code], ((int)(numcodes * columns / 4), rows-10),cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0))
                    result[numcodes] = code
                    print("code", code)
                    numcodes += 1
                    if numcodes >= 3:
                        found = True
                        break
                numconti = 0
                sumcodes = [0]*res_columns
            elif coldet[x] >= 0:
                if numconti > 10:
                    sumcodes[numconti - 10] = coldet[x]
                if numconti == 10:
                    pass
                    
                numconti += 1
            else:
                numconti = 0

        
        print("result",result)
        final_result = result[0] * 10 + result[1]
        final_result *= math.pow(10, result[2])
        resistance = final_result
        unit = " Ohm"
        if resistance >= 1000.0:
            resistance /= 1000
            unit = "k Ohm"
        if resistance >= 1000000.0:
            resistance /= 1000000
            unit = "m Ohm"

        answer = f'{resistance:.1f}' + unit
        cv.putText(image, answer, (50, 50),cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0))
        
        print("Resistance : ",answer)

    final = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cv.imshow("Final Output", final)
    cv.waitKey(0)
