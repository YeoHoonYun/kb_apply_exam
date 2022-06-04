# # 3. OCR
# 다음 이미지에서 체크박스를 찾고 체크 유무를 판단할 수 있는 처리 방법을 제시하고 구현합니다.
# •	처리방법에 대한 방안은 2장 내로 기술하며 (markup format) 작동하는 코드 또는 이미지를 제출합니다.
# •	공개된 모델 또는 오픈소스 사용시 반드시 그 출처를 명시 및 첨부하며 활용한 방법을 기술합니다.
# 1. 아래 문서 이미지에서 체크박스를 찾고 체크 유무를 판별하는 프로그램을 구현합니다.
#
# 2. 아래 문서 이미지에서 체크박스를 찾고 체크 유무를 판별하는 프로그램을 구현합니다.
#
# 테이블 인식 및 처리
# 다음 이미지에서 테이블의 구조를 추출하고 연관관계를 도출할 수 있는 방법을 제시하고 구현합니다.
# •	OCR이 되어있다 가정합니다 (문자 위치 및 문자 인식 된 상태).
# •	처리방법에 대한 방안은 2장 내로 기술하며 (markup format) 가능하다면 작동하는 코드 또는 이미지를 제출합니다.
# •	공개된 모델 또는 오픈소스 사용시 반드시 그 출처를 명시 및 첨부하며 활용한 방법을 기술합니다.
# 1. 아래 문서 이미지에서 테이블 구조 (영역) 을 인지하고 구조간 관계를 설명할 수 있는 프로그램을 구현합니다.
# 예) 구분-개인영역, 질문-발열과 호흡기 증상, 답변-예, 개인영역-발열과 호흡기 증상-예
#
# 2. 아래 문서 이미지에서 테이블 구조 (영역) 을 인지하고 구조간 관계를 설명할 수 있는 프로그램을 구현합니다.
# 예) 시각-1-아이가 눈을 잘 맞춥니까?-예

# conda install -c conda-forge poppler
# pip install pdf2image
from pdf2image import convert_from_path
import cv2, numpy

# 1-1)금융거래목적확인서.pdf
images = convert_from_path("data\\03. OCR\\1.pdf")
img = cv2.cvtColor(numpy.array(images[0]), cv2.COLOR_BGR2GRAY)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 25)

# 금융거래 목적 (start x y / end x y 136 43)
train1_1 = img[321:364,394:530].copy()
train1_2 = img[368:411,393:529].copy()
train1_3 = img[414:457,393:529].copy()
train1_4 = img[458:501,393:529].copy()
train1_5 = img[321:364,889:1025].copy()

train1_1_list = train1_1.reshape(1, -1)[0]
train1_2_list = train1_2.reshape(1, -1)[0]
train1_3_list = train1_3.reshape(1, -1)[0]
train1_4_list = train1_4.reshape(1, -1)[0]
train1_5_list = train1_5.reshape(1, -1)[0]

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
import numpy as np
X = np.array([train1_1_list, train1_2_list, train1_3_list, train1_4_list, train1_5_list])
Y = np.array([1, 0, 0, 0, 0])
# kmeans_1 = KMeans(n_clusters=2, random_state=0).fit(X)
# kmeans_1 = LogisticRegression(random_state=0).fit(X,Y)
# kmeans_1 = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(X,Y)
kmeans_1 = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(X)
# kmeans_1 = RandomForestClassifier().fit(X,Y)

test1_1 = img[323:366,393:529].copy()
test1_2 = img[370:413,393:529].copy()
test1_3 = img[416:459,393:529].copy()
test1_4 = img[460:503,393:529].copy()
test1_5 = img[323:366,888:1024].copy()

test1_1_list = test1_1.reshape(1, -1)[0]
test1_2_list = test1_2.reshape(1, -1)[0]
test1_3_list = test1_3.reshape(1, -1)[0]
test1_4_list = test1_4.reshape(1, -1)[0]
test1_5_list = test1_5.reshape(1, -1)[0]

# print(kmeans_1.predict([test1_1_list, test1_2_list, test1_3_list, test1_4_list, test1_5_list]))
print(kmeans_1.fit_predict([test1_1_list, test1_2_list, test1_3_list, test1_4_list, test1_5_list]))

#고객확인사항
train2_1 = img[723:760,147:245].copy()
train2_2 = img[723:760,239:337].copy()
train2_3 = img[723:760,331:429].copy()
train2_4 = img[723:760,447:545].copy()
train2_5 = img[723:760,598:696].copy()
train2_6 = img[723:760,774:872].copy()
train2_7 = img[723:760,916:1014].copy()
train2_8 = img[723:760,1028:1126].copy()
train2_9 = img[723:760,1218:1316].copy()
train2_10 = img[723:760,1412:1510].copy()

train2_1_list = train2_1.reshape(1, -1)[0]
train2_2_list = train2_2.reshape(1, -1)[0]
train2_3_list = train2_3.reshape(1, -1)[0]
train2_4_list = train2_4.reshape(1, -1)[0]
train2_5_list = train2_5.reshape(1, -1)[0]
train2_6_list = train2_6.reshape(1, -1)[0]
train2_7_list = train2_7.reshape(1, -1)[0]
train2_8_list = train2_8.reshape(1, -1)[0]
train2_9_list = train2_9.reshape(1, -1)[0]
train2_10_list = train2_10.reshape(1, -1)[0]

X = np.array([train2_1_list,train2_2_list,train2_3_list,train2_4_list,train2_5_list,train2_6_list,train2_7_list,train2_8_list,train2_9_list,train2_10_list])
Y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
# kmeans_2 = KMeans(n_clusters=3, random_state=0).fit(X)
# kmeans_2 = LogisticRegression(random_state=0).fit(X,Y)
# kmeans_2 = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(X,Y)
kmeans_2 = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(X)
# kmeans_2 = RandomForestClassifier().fit(X,Y)

test2_1 = img[723:760,145-5:243-5].copy()
test2_2 = img[723:760,237-5:335-5].copy()
test2_3 = img[723:760,329-5:427-5].copy()
test2_4 = img[723:760,445-5:543-5].copy()
test2_5 = img[723:760,596-5:694-5].copy()
test2_6 = img[723:760,772-5:870-5].copy()
test2_7 = img[723:760,914-5:1012-5].copy()
test2_8 = img[723:760,1026-5:1124-5].copy()
test2_9 = img[723:760,1216-5:1314-5].copy()
test2_10 = img[723:760,1410-5:1508-5].copy()

test2_1_list = test2_1.reshape(1, -1)[0]
test2_2_list = test2_2.reshape(1, -1)[0]
test2_3_list = test2_3.reshape(1, -1)[0]
test2_4_list = test2_4.reshape(1, -1)[0]
test2_5_list = test2_5.reshape(1, -1)[0]
test2_6_list = test2_6.reshape(1, -1)[0]
test2_7_list = test2_7.reshape(1, -1)[0]
test2_8_list = test2_8.reshape(1, -1)[0]
test2_9_list = test2_9.reshape(1, -1)[0]
test2_10_list = test2_10.reshape(1, -1)[0]

# print(kmeans_2.predict([test2_1_list,test2_2_list,test2_3_list,test2_4_list,test2_5_list,test2_6_list,test2_7_list,test2_8_list,test2_9_list,test2_10_list]))
print(kmeans_2.fit_predict([test2_1_list,test2_2_list,test2_3_list,test2_4_list,test2_5_list,test2_6_list,test2_7_list,test2_8_list,test2_9_list,test2_10_list]))

train3_1 = img[875:912,1298:1396].copy()
train3_2 = img[875:912,1425:1523].copy()
train3_3 = img[1009:1046,1298:1396].copy()
train3_4 = img[1009:1046,1425:1523].copy()
train3_5 = img[1140:1177,1298:1396].copy()
train3_6 = img[1140:1177,1425:1523].copy()

train3_1_list = train3_1.reshape(1, -1)[0]
train3_2_list = train3_2.reshape(1, -1)[0]
train3_3_list = train3_3.reshape(1, -1)[0]
train3_4_list = train3_4.reshape(1, -1)[0]
train3_5_list = train3_5.reshape(1, -1)[0]
train3_6_list = train3_6.reshape(1, -1)[0]

# X = np.array([train3_1_list,train3_2_list,train3_3_list,train3_4_list,train3_5_list,train3_6_list])
X = np.array([train2_1_list,train2_2_list,train2_3_list,train2_4_list,train2_5_list,train2_6_list,train2_7_list,train2_8_list,train2_9_list,train2_10_list,train3_1_list,train3_2_list,train3_3_list,train3_4_list,train3_5_list,train3_6_list])
Y = [1,0,0,1,0,0,0,0,0,0,2,1,0,1,0,0]
# kmeans_3 = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward').fit(X)
kmeans_3 = RandomForestClassifier().fit(X,Y)

test3_1 = img[875:912,1298-5:1396-5].copy()
test3_2 = img[875:912,1425-5:1523-5].copy()
test3_3 = img[1009:1046,1298-5:1396-5].copy()
test3_4 = img[1009:1046,1425-5:1523-5].copy()
test3_5 = img[1140:1177,1298-5:1396-5].copy()
test3_6 = img[1140:1177,1425-5:1523-5].copy()

test3_1_list = test3_1.reshape(1, -1)[0]
test3_2_list = test3_2.reshape(1, -1)[0]
test3_3_list = test3_3.reshape(1, -1)[0]
test3_4_list = test3_4.reshape(1, -1)[0]
test3_5_list = test3_5.reshape(1, -1)[0]
test3_6_list = test3_6.reshape(1, -1)[0]

# print(kmeans_3.fit_predict([test3_1_list, test3_2_list, test3_3_list, test3_4_list, test3_5_list, test3_6_list]))
print(kmeans_3.predict([test3_1_list, test3_2_list, test3_3_list, test3_4_list, test3_5_list, test3_6_list]))

# cv2.imshow('image',test3_4)
# cv2.waitKey(0)
# cv2.imwrite('1.jpg', img)

# # 1-2)대출거래약정서.pdf
# images = convert_from_path("data\\03. OCR\\1-2.pdf")
# img = cv2.cvtColor(numpy.array(images[0]), cv2.COLOR_RGB2BGR)
# cv2.imshow('image',img)
# cv2.waitKey(0)
#
# # 2-1)신규거래신청서.pdf
# images = convert_from_path("data\\03. OCR\\2-1.pdf")
# img = cv2.cvtColor(numpy.array(images[0]), cv2.COLOR_RGB2BGR)
# cv2.imshow('image',img)
# cv2.waitKey(0)
#
# # 2-2)위임장.pdf
# images = convert_from_path("data\\03. OCR\\2-2.pdf")
# img = cv2.cvtColor(numpy.array(images[0]), cv2.COLOR_RGB2BGR)
# cv2.imshow('image',img)
# cv2.waitKey(0)
