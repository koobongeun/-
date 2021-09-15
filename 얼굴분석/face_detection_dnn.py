import cv2
import numpy as np

#caffemodel를 불러옴
model_name = 'C:/computervision/model/res10_300x300_ssd_iter_140000.caffemodel'
#input, layer 등 모델 구조에 대한메타 정보가 저장 (caffe 모델을 프로그램에서 사용하기 위해서는 두파일을 가져와서 사용)
prototxt_name = 'C:/computervision/model/deploy.prototxt.txt'
min_confidence = 0.15
file_name = "C:/computervision/image/aaaa.jpg"

def detectAndDisplay(frame):
    #모델 읽어오기
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    #이미지를 blob형식으로 불러와서 모델이 분석하기 편하게 resize 진행
    #image: 사용할 이미지를 정해주는 것이고 여기서는 모델에서 300x300의 크기를 사용
    #scalefactor: 이미지의 크기비율을 지정하는 것으로 여기서는 1.0 변형 없음
    #size: cnn에서 사용할 이미지 크기를 지정해 주는 것으로 300x300을 사용
    #mean: RGB값의 일부를 제외해서 dnn이 분석하기 쉽게 단순화해주는 값/ noise제거
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #caffe 모델이 처리하고 결과값을 detections라는 이름의 배열에 저장
    model.setInput(blob)
    detections = model.forward()

    #4차원 배열인 detections를 2차원 배열로 만듦 >> 필요한 내용만 가져오기 위해 단순화 과정
    print(detections[0, 0])
    #결과값:200 >> 모델이 가져올수있는 최대 이미지 박스 크기
    print(detections.shape[2])

    #detections의 값들을 반복해서 구함
    for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > min_confidence:
                    # 3:7 = 전체 이미지의 폭, 높이 얼굴의 폭, 높이 이것을 곱해주면 좌표 생성
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    # 곱이 소수점이니깐 타입을 정수형을 바꿔줌
                    (startX, startY, endX, endY) = box.astype("int")
                    print(i, confidence,detections[0, 0, i, 3], startX, startY, endX, endY)
     
                    text = "{:.2f}%".format(confidence * 100)
                    #글씨가 써지는 부분 표시
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Face Detection by dnn", frame)
    
    
print("OpenCV version:")
print(cv2.__version__)
 
img = cv2.imread(file_name)
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

(height, width) = img.shape[:2]

cv2.imshow("Original Image", img)

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
