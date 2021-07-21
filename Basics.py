import cv2
import numpy as np
import face_recognition
from PIL import Image

# Face Recongition : 영상으로 부터 사람의 얼굴 부분을 검출
# Face Identification : 검출된 얼굴을 기존 학습된 데이터를 통해 누구인지 파악
# reference
# 1. CV ZONE
# 2. https://github.com/jeonggunlee/faceid

# First STEP
# 이미지를 가져오기 형식 (폴더/이미지이름.jpg)
imgElon = face_recognition.load_image_file('set/Elon-Musk.jpg')
# 이미지를 가져왔으면 RGB로 변환하기
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

# Test image
imgTest = face_recognition.load_image_file('set/Eleon.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Test image2
imgTest2 = face_recognition.load_image_file('set/Certificate.jpg')
imgTest2 = cv2.cvtColor(imgTest2, cv2.COLOR_BGR2RGB)

# Test
#img_color = cv2.imread('set/Jeong.jpg')
# imgTest3 = cv2.resize(img_color, (680, 480), interpolation=cv2.INTER_AREA)
# imgTest3 = face_recognition.load_image_file('set/Jeong.jpg')
imgTest3 = face_recognition.load_image_file('set/Jeong.png')
imgTest3 = cv2.cvtColor(imgTest3, cv2.COLOR_BGR2RGB)

# Second STEP
# 사진에서 얼굴 위치 찾기 함수 face_locations
faceLoc = face_recognition.face_locations(imgElon)[0]
print(faceLoc) #-> pixel 좌표로 나옴 -> 얼굴 사각형 만들기 할때 좌표 얻음
# Elon-Musk,jpg : 660x 495 pixel
# faceLoc = [top, right, bottom, left]
encodeElon = face_recognition.face_encodings(imgElon)[0]
# print(encodeElon) 사진의 얼굴을 데이터화하는 작업 : encoding

# 사각형 ( 객체, 첫번째 좌표, 두번째 좌표, 색깔, 두께) / 왼쪽 위(296,168), 오른쪽 아래(425, 297)
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255), 2)
# cv2.rectangle(imgElon, (296,168), (425, 297), (255,0,255), 2)

# TEST
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0 ,255), 2)

# TEST2
faceLocTest2 = face_recognition.face_locations(imgTest2)[0]
encodeTest2 = face_recognition.face_encodings(imgTest2)[0]
cv2.rectangle(imgTest2, (faceLocTest2[3], faceLocTest2[0]), (faceLocTest2[1], faceLocTest2[2]), (255, 0 ,255), 2)

# TEST3
faceLocTest3 = face_recognition.face_locations(imgTest3)[0]
encodeTest3 = face_recognition.face_encodings(imgTest3)[0]
cv2.rectangle(imgTest3, (faceLocTest3[3], faceLocTest3[0]), (faceLocTest3[1], faceLocTest3[2]), (255, 0 ,255), 2)

# Elon 사진과 Test Elon 사진을 비교했을때 동일 인물인가?
# 결과 값이 True 면 동일 인물
results = face_recognition.compare_faces([encodeElon], encodeTest)
results2 = face_recognition.compare_faces([encodeTest2], encodeTest3)
results3 = face_recognition.compare_faces([encodeElon], encodeTest3)
# print(results) True
# print(results2) True
# print(results3) False

# 얼만큼 차이나는지 수치화해서 보여줌 작으면 작을 수 록 좋음
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
faceDis2 = face_recognition.face_distance([encodeTest2], encodeTest3)
print(results, faceDis)
print(results2, faceDis2)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.putText(imgTest2, f'{results2} {round(faceDis2[0], 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)



# imshow(Windowname, img object)
cv2.imshow('Elon-Musk', imgElon)
cv2.imshow('Eleon', imgTest)
cv2.imshow('MeFirst', imgTest2)
cv2.imshow('Mesecond', imgTest3)
cv2.waitKey(0)

