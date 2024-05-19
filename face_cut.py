import cv2

# Haar Cascade 파일 경로
cascade_path = './haarcascade_frontalface_default.xml'

# 이미지 읽기
image_path = './image/로켓펀치 소희/로켓펀치 소희_7.jpg'
image = cv2.imread(image_path)

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 검출용 CascadeClassifier 객체 생성
face_cascade = cv2.CascadeClassifier(cascade_path)

# 얼굴 검출
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=5, minSize=(30, 30))

# 검출된 얼굴 주변에 사각형 그리기
for (x, y, w, h) in faces:
    # 녹색으로 사각형 바운더리 그리기
    # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    detected_face = image[y:y+h, x:x+w]
    cv2.imwrite('./detected_face1.jpg', detected_face)

# 결과 이미지 출력
# cv2.imshow('Detected Faces', image)
# cv2.waitKey(0)

# 결과 이미지 저장
output_path = './detected_faces1.jpg'
cv2.imwrite(output_path, image)

# cv2.destroyAllWindows()
