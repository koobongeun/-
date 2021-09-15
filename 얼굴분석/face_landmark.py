import numpy as np
import dlib
import cv2

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

predictor_file = 'C:/computervision/model/shape_predictor_68_face_landmarks.dat'
image_file = 'C:/computervision/image/marathon_03.jpg'

#정면사진을 detector
detector = dlib.get_frontal_face_detector()
#얼굴에서 찾아낸 것을 predictor 변수안에 넣는다.
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
#인식률 향상
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#얼굴인식
rects = detector(gray, 1)
print("Number of faces detected: {}".format(len(rects)))

#얼굴인식한것을 처리
for (i, rect) in enumerate(rects):
    # 얼굴의 각 부분을 추출하고 점들의 x좌표와 y좌표를 하나의 배열로 만들어줌
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    #모든 점을 가져옴
    show_parts = points[EYES]
    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        cv2.putText(image, "{}".format(i + 1), (x, y - 2),
		cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

cv2.imshow("Face Landmark", image)
cv2.waitKey(0)   
