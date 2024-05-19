import os
import cv2
import time

# Haar Cascade 파일 경로
cascade_path = './haarcascade_frontalface_default.xml'


# 입력 디렉토리 경로
input_path = './image'

# 출력 디렉토리 경로
output_path = './detected_face'

save_count = []

# 입력 폴더 순회
for folder in os.listdir(input_path):
    input_dir = os.path.join(input_path, folder)
    
    # 디렉토리인지 확인
    if os.path.isdir(input_dir):
        # 출력 디렉토리 경로 생성
        output_dir = os.path.join(output_path, folder)
        
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total_cuted_image = 0

        # 입력 디렉토리의 이미지 파일들을 순회하면서 얼굴 검출
        for filename in os.listdir(input_dir):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                image_path = os.path.join(input_dir, filename)
                
                try:
                    # 이미지 읽기
                    image = cv2.imread(image_path)

                    # 그레이스케일 변환
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # 얼굴 검출용 CascadeClassifier 객체 생성
                    face_cascade = cv2.CascadeClassifier(cascade_path)

                    # 얼굴 검출
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=5, minSize=(30, 30))

                    print(image_path)

                    for (x, y, w, h) in faces:
                        detected_face = image[y:y+h, x:x+w]
                        output_file_path = os.path.join(output_dir, filename)
                        cv2.imwrite(output_file_path, detected_face)
                    total_cuted_image += 1
                        
                except Exception as e:
                    print(f"Error processing image {image_path}: {str(e)}")
                    continue
                
        save_count.append({input_dir : total_cuted_image})

# 완료 메시지 출력
print("---------------------------------")
print("얼굴 검출 및 저장이 완료되었습니다.")

for item in save_count:
    for key, value in item.items():
        print(f"{key}: {value}")
print("---------------------------------")