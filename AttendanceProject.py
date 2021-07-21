import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# 출석 시간을 알기 위해 datetime

# from PIL import Image

# Face Recongition : 영상으로 부터 사람의 얼굴 부분을 검출
# Face Identification : 검출된 얼굴을 기존 학습된 데이터를 통해 누구인지 파악
# reference
# 1. CV ZONE
# 2. https://github.com/jeonggunlee/faceid

# 경로 설정
path = 'set'
# 이미지 넣을 리스트 초기화
images = []
# 출석부 이름 넣을 리스트 초기화
classNames = []
# 경로에 있는 이미지들 이름 불러오기 확장명도 같이 붙어있음 jpg, png
myList = os.listdir(path)
print(myList)

# 사진 수 많큼 이미지랑 이름을 넣어야함
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')  # 경로에 있는 이미지 파일들을 컬러로 읽음
    images.append(curImg)  # 읽은 컬러 이미지들을 앞서 초기화한 리스트에 넣음 append 함수
    classNames.append(os.path.splitext(cls)[0])  # 이미지들의 이름을 확장자명을 빼고 저장
print(classNames)

# encoding은 이미지를 데이터화 하는것
def findEncodings(images):
    encodeList = []
    for img in images: # 읽은 컬러 이미지들을 RGB 형태로 바꿈
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0] # RGB 이미지들을 데이터 화
        encodeList.append(encode) # 데이터화한 이미지들을 리스트에 순서대로 넣음
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f: # 파일 열기
        myDataList = f.readlines() # 열어 놓은 파일 한줄씩 읽기
        nameList = []
        #print(myDataList)
        for line in myDataList: # 읽은 데이터 리스트
            entry = line.split(',') # ,를 기준으로 이름과 시간을 나눔 [name, time]
            nameList.append(entry[0]) # entry[0]은 이름이니까 초기화해둔 nameList에 출석한 name을 추가
        if name not in nameList: # nameList에 현재 이미지의 이름이 없다면? 추가(중복 고려한것)
            now = datetime.now() # 현재 시간
            dtString = now.strftime('%H:%M:%S') # 시간, 분, 초
            f.writelines(f'\n{name},{dtString}') # 이름, 시간, 분, 초 쓰는 함수





encodeListKnown = findEncodings(images)
print('Encoding Complete')

# 영상에서 얼굴 인식하기 위해 영상을 카메라로 찍음
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read() # 카메라에 영상 이미지들을 읽어들임
    imgS = cv2.resize(img,(0,0), None,0.25,0.25) # 카메라의 영상 이미지 사이즈를 0.25배만큼 줄임
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # 읽어들인 이미지를 RGB형태로 변환

    facesCurFrame = face_recognition.face_locations(imgS) # 읽어들인 이미지들의 모든 얼굴 좌표를 저장
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame) # RGB 형태로 바꾼 이미지의 얼굴 위치에 있는 얼굴을 데이터화
    # zip 함수는 encodesCurFrame은 encodeFace에 차례로, facesCurFrame은 faceLoc에 차례로 (encodeFace, faceLoc)으로 행렬로 만들어줌
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        # 현재 영상 이미지와 데이터 셋의 이미지를 비교하기 위해 가장 알맞는 사람을 찾음
        # faceDis가 가장 낮은 값이 가장 매칭이 잘 맞다는 말임
        matchIndex = np.argmin(faceDis)
        # matches[True or False]일텐데 실제 있는 사람이라면 그때 이름은 누구인지 대문자로 출력
        if matches[matchIndex]:
            name = classNames[matchIndex].upper() # upper는 대문자로 출력하는 함수
            #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2) # 빈 사각형
            cv2.rectangle(img, (x1,y2-35), (x2, y2), (0,255,0), cv2.FILLED) # 채워진 사각형
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

# faceLoc = face_recognition.face_locations(imgElon)[0]
# faceLoc = [top, right, bottom, left]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255), 2)
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceL
# results = face_recognition.compare_faces([encodeElon], encodeTest)
# faceDis = face_recognition.face_distance([encodeElon], encodeTest)