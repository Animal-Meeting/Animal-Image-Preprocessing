import os
import cv2

# Haar Cascade 파일 경로
cascade_path = './haarcascade_frontalface_default.xml'

# 얼굴 검출용 CascadeClassifier 객체 생성
face_cascade = cv2.CascadeClassifier(cascade_path)

# 입력 디렉토리 경로
input_root = './image'

# 출력 디렉토리 경로
output_root = './detected_face'

# 입력 폴더 순회
for folder in os.listdir(input_root):
    input_dir = os.path.join(input_root, folder)
    
    # 디렉토리인지 확인
    if os.path.isdir(input_dir):
        # 출력 디렉토리 경로 생성
        output_dir = os.path.join(output_root, folder)
        
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 입력 디렉토리의 이미지 파일들을 순회하면서 얼굴 검출
        for filename in os.listdir(input_dir):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                image_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                
                # 이미지 읽기
                image = cv2.imread(image_path)

                # 그레이스케일 변환
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # 얼굴 검출
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=9, minSize=(30, 30))

                # 검출된 얼굴 주변에 사각형 그리기
                for (x, y, w, h) in faces:
                    # 검출된 얼굴 이미지만 추출하여 저장
                    detected_face = image[y:y+h, x:x+w]
                    cv2.imwrite(output_path, detected_face)

# 완료 메시지 출력
print("얼굴 검출 및 저장이 완료되었습니다.")