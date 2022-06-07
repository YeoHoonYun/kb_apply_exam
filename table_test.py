import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

from pdf2image import convert_from_path
import cv2, numpy

images = convert_from_path("data\\03. OCR\\1.pdf")
# 처리한 이미지를 흑백으로 전환
img = cv2.cvtColor(numpy.array(images[0]), cv2.COLOR_BGR2GRAY)
# 이미지를 작은 영역별로 Thresholding
# cv2.ADAPTIVE_THRESH_MEAN_C : 주변영역의 평균값으로 결정
# threshold type
# blockSize – thresholding을 적용할 영역 사이즈
# C – 평균이나 가중평균에서 차감할 값
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)

# 고정된 임계값을 설정하고 결과 출력
# src – input image로 single-channel 이미지.(grayscale 이미지)
# thresh – 임계값
# maxval – 임계값을 넘었을 때 적용할 value
# type – thresholding type
thresh, img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bin = 255-img_bin
kernel_len = np.array(img).shape[1]//100

# 수직
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# 수평
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# 특정 구조화 요소를 사용하여 이미지를 침식
# iterations 침식을 반복할 횟수 지정
image_1 = cv2.erode(img_bin, ver_kernel, iterations=2)
# 특정 구조화 요소를 사용해서 이미지를 팽창
# iterations 팽창을 반복할 횟수 지정
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=2)

image_2 = cv2.erode(img_bin, hor_kernel, iterations=2)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=2)

# cv2.addWeighted 가중치 합, 평균 연산
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
# iterations 침식을 반복할 횟수 지정
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
# 고정된 임계값을 설정하고 결과 출력
# src – input image로 single-channel 이미지.(grayscale 이미지)
# thresh – 임계값
# maxval – 임계값을 넘었을 때 적용할 value
# type – thresholding type
thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#비트 연산 xor, not
# bitxor = cv2.bitwise_xor(img,img_vh)
# bitnot = cv2.bitwise_not(bitxor)

# 외곽선 정보 검출
# image, mode, method
# image: 입력 영상. non-zero 픽셀을 객체로 간주함.
# mode: 외곽선 검출 모드. cv2.RETR_로 시작하는 상수. RETR_TREE 계층형 데이터 출력
# method: 외곽선 근사화 방법. cv2.CHAIN_APPROX_로 시작하는 상수.
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

# 검출된 윤곽선들을 소팅함.
contours, boundingBoxes = sort_contours(contours, "top-to-bottom")
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
mean = np.mean(heights)

box = []
first_box = []
second_box = []
third_box = []
forth_box = []

for c in contours:
    # contour의 외접하는 똑바로 세워진 사각형의 좌표를 얻음.
    x, y, w, h = cv2.boundingRect(c)
    box.append([x, y, w, h])

    # pytesseract.image_to_string(img[y:y+h, x:x+w].copy(), lang="kor")

    # 금융거래목적 셀렉트
    if (w < 10 or h < 10):
        continue

    if ((y>=300 and y<=500) and (w>=135 and w<=140) and (h>=40 and h<=45)):
        first_box.append([x,y,w,h]) # 기타# 급여# 공과금# 모임# 상거래
    elif ((y >= 720 and y <= 788) and h<=50):
        second_box.append([x, y, w, h])
    elif ((y >= 788 and y <= 1400) and (w>=100 and w<=130) and (h>=100 and h<=150)):
        third_box.append([x, y, w, h])
    elif (y >= 1300 and y <= 1500) and (w >= 100 and w <= 200):
        forth_box.append([x, y, w, h])
    elif (y >= 1550 and y <= 1700) and (w >= 100 and w <= 200):
        forth_box.append([x, y, w, h])
        # print([x, y, w, h])
        # plotting = plt.imshow(img[y:y + h, x:x + w], cmap='gray')
        # plt.show()

print(len(first_box), len(second_box), len(third_box), len(forth_box))
for i, v in enumerate(first_box):
    slice_img = img[v[1]:v[1] + v[3], v[0]:v[0] + v[2]].copy()
    cv2.imwrite("result/0/%s.jpg" % i, slice_img)
for i, v in enumerate(second_box):
    slice_img = img[v[1]:v[1] + v[3], v[0]:v[0] + v[2]].copy()
    cv2.imwrite("result/1/%s.jpg" % i, slice_img)
for i, v in enumerate(third_box):
    slice_img = img[v[1]:v[1] + v[3], v[0]:v[0] + v[2]].copy()
    cv2.imwrite("result/2/%s.jpg" % i, slice_img)
for i, v in enumerate(forth_box):
    slice_img = img[v[1]:v[1]+v[3], v[0]:v[0]+v[2]].copy()
    cv2.imwrite("result/3/%s.jpg" % i, slice_img)
    # cv2.imshow('image',slice_img)
    # cv2.waitKey(0)

# #Creating two lists to define row and column in which cell is located
# row=[]
# column=[]
# j=0
# #Sorting the boxes to their respective row and column
# for i in range(len(box)):
#     if(i==0):
#         column.append(box[i])
#         previous=box[i]
#     else:
#         if(box[i][1]<=previous[1]+mean/2):
#             column.append(box[i])
#             previous=box[i]
#             if(i==len(box)-1):
#                 row.append(column)
#         else:
#             row.append(column)
#             column=[]
#             previous = box[i]
#             column.append(box[i])
# print(column)
# print(row)
#
# #calculating maximum number of cells
# countcol = 0
# for i in range(len(row)):
#     countcol = len(row[i])
#     if countcol > countcol:
#         countcol = countcol
#
# #Retrieving the center of each column
# center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
# center=np.array(center)
# center.sort()
#
# #Regarding the distance to the columns center, the boxes are arranged in respective order
# finalboxes = []
# for i in range(len(row)):
#     lis=[]
#     for k in range(countcol):
#         lis.append([])
#     for j in range(len(row[i])):
#         diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
#         minimum = min(diff)
#         indexing = list(diff).index(minimum)
#         lis[indexing].append(row[i][j])
#     finalboxes.append(lis)
#
# # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
# outer = []
# for i in range(len(finalboxes)):
#     for j in range(len(finalboxes[i])):
#         inner =''
#         if (len(finalboxes[i][j]) == 0):
#             outer.append(' ')
#         else:
#             for k in range(len(finalboxes[i][j])):
#                 y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
#                              finalboxes[i][j][k][3]
#                 finalimg = bitnot[x:x + h, y:y + w]
#                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
#                 border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
#                 resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#                 dilation = cv2.dilate(resizing, kernel, iterations=1)
#                 erosion = cv2.erode(dilation, kernel, iterations=1)

                # plotting = plt.imshow(erosion, cmap='gray')
                # plt.show()
#                 out = pytesseract.image_to_string(erosion, lang="kor")
#                 if (len(out) == 0):
#                     out = pytesseract.image_to_string(erosion, lang="kor", config='--psm 3')
#                 inner = inner + " " + out
#             outer.append(inner)
#
# #Creating a dataframe of the generated OCR list
# arr = np.array(outer)
# dataframe = pd.DataFrame(arr.reshape(len(row),countcol))
# print(dataframe)
# data = dataframe.style.set_properties(align="left")
# #Converting it in a excel-file
# data.to_excel('output.xlsx')
