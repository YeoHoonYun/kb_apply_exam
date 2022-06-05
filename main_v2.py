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
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, fbeta_score, recall_score, f1_score
import numpy as np

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.reshape(1, -1)[0]

# 1-1)금융거래목적확인서.pdf
images = sorted(glob.glob('augumentation/0/*.jpg'))
data = [read_img(x) for x in images]
target = [x.split("_")[-1][0] for x in images]
print("데이터 예시 : ",data)
print("타겟 예시 : ", target)
test_size = 0.8

x_train, x_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    stratify=target,
                                                    random_state=34)
X = np.array(x_train)
Y = np.array(y_train)

model = LogisticRegression(random_state=0).fit(X,Y)
y_pred = model.predict(x_test)

print("1. 금융거래목적 정확도")
print(confusion_matrix(y_test, y_pred))
print("accuracy : ", accuracy_score(y_test, y_pred))
# print('Macro average precision : ', precision_score(y_test, y_pred, average='macro'))
# print('Micro average precision : ', precision_score(y_test, y_pred, average='micro'))
# print('Macro average recall : ', recall_score(y_test, y_pred, average='macro'))
# print('Micro average recall : ', recall_score(y_test, y_pred, average='micro'))
# print('Macro average fbeta-score : ', fbeta_score(y_test, y_pred, beta=1, average='macro'))
# print('Micro average fbeta-score : ', fbeta_score(y_test, y_pred, beta=1, average='micro'))
# print('Macro average f1-score : ', f1_score(y_test, y_pred, average='macro'))
# print('Micro average f1-score : ', f1_score(y_test, y_pred, average='micro'))
print()

images = sorted(glob.glob('augumentation/1/*.jpg'))
data = [read_img(x) for x in images]
target = [x.split("_")[-1][0] for x in images]

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, shuffle=True, stratify=target, random_state=34)
X = np.array(x_train)
Y = np.array(y_train)

model = LogisticRegression(random_state=0).fit(X,Y)
y_pred = model.predict(x_test)

print("2. 고객확인사항 정확도")
print(confusion_matrix(y_test, y_pred))
print("accuracy : ", accuracy_score(y_test, y_pred))
# print('Macro average precision : ', precision_score(y_test, y_pred, average='macro'))
# print('Micro average precision : ', precision_score(y_test, y_pred, average='micro'))
# print('Macro average recall : ', recall_score(y_test, y_pred, average='macro'))
# print('Micro average recall : ', recall_score(y_test, y_pred, average='micro'))
# print('Macro average fbeta-score : ', fbeta_score(y_test, y_pred, beta=1, average='macro'))
# print('Micro average fbeta-score : ', fbeta_score(y_test, y_pred, beta=1, average='micro'))
# print('Macro average f1-score : ', f1_score(y_test, y_pred, average='macro'))
# print('Micro average f1-score : ', f1_score(y_test, y_pred, average='micro'))
print()

images = sorted(glob.glob('augumentation/2/*.jpg'))
data = [read_img(x) for x in images]
target = [x.split("_")[-1][0] for x in images]

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, shuffle=True, stratify=target, random_state=34)
X = np.array(x_train)
Y = np.array(y_train)

model = LogisticRegression(random_state=0).fit(X,Y)
mapping = {
    "0" : "0",
    "1" : "1",
    "2" : "0"
}

y_pred = [mapping[x] for x in model.predict(x_test)]
y_test = [mapping[x] for x in y_test]

print("3. 공통 정확도")
print(confusion_matrix(y_test, y_pred))
print("accuracy : ", accuracy_score(y_test, y_pred))
# print('Macro average precision : ', precision_score(y_test, y_pred, average='macro'))
# print('Micro average precision : ', precision_score(y_test, y_pred, average='micro'))
# print('Macro average recall : ', recall_score(y_test, y_pred, average='macro'))
# print('Micro average recall : ', recall_score(y_test, y_pred, average='micro'))
# print('Macro average fbeta-score : ', fbeta_score(y_test, y_pred, beta=1, average='macro'))
# print('Micro average fbeta-score : ', fbeta_score(y_test, y_pred, beta=1, average='micro'))
# print('Macro average f1-score : ', f1_score(y_test, y_pred, average='macro'))
# print('Micro average f1-score : ', f1_score(y_test, y_pred, average='micro'))
print()

images = sorted(glob.glob('augumentation/3/*.jpg'))
data = [read_img(x) for x in images]
target = [x.split("_")[-1][0] for x in images]

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, shuffle=True, stratify=target, random_state=34)
X = np.array(x_train)
Y = np.array(y_train)

model = LogisticRegression(random_state=0).fit(X,Y)
y_pred = model.predict(x_test)

print("4. 고객 확인 정확도")
print(confusion_matrix(y_test, y_pred))
print("accuracy : ", accuracy_score(y_test, y_pred))
# print('Macro average precision : ', precision_score(y_test, y_pred, average='macro'))
# print('Micro average precision : ', precision_score(y_test, y_pred, average='micro'))
# print('Macro average recall : ', recall_score(y_test, y_pred, average='macro'))
# print('Micro average recall : ', recall_score(y_test, y_pred, average='micro'))
# print('Macro average fbeta-score : ', fbeta_score(y_test, y_pred, beta=1, average='macro'))
# print('Micro average fbeta-score : ', fbeta_score(y_test, y_pred, beta=1, average='micro'))
# print('Macro average f1-score : ', f1_score(y_test, y_pred, average='macro'))
# print('Micro average f1-score : ', f1_score(y_test, y_pred, average='micro'))
