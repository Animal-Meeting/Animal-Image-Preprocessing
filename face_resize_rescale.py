import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

# 입력 디렉토리 경로
input_path = './detected_face'

# 출력 디렉토리 경로
output_path = './resize_rescaled_images'

# 이미지 크기 설정 (예: 256x256)
img_width, img_height = 256, 256

# ImageDataGenerator를 사용하여 이미지 리스케일링 설정
datagen = ImageDataGenerator(
    rescale=1.0/255.0  # 0-1 사이 값으로 리스케일링
)

# 출력 디렉토리가 없으면 생성
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 입력 디렉토리의 이미지 파일들을 순회하면서 리사이즈 및 리스케일링
for folder in os.listdir(input_path):
    input_dir = os.path.join(input_path, folder)
    
    print("Start :" + input_dir)
    
    # 디렉토리인지 확인
    if os.path.isdir(input_dir):
        # 출력 디렉토리 경로 생성
        output_dir = os.path.join(output_path, folder)
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 입력 디렉토리의 이미지 파일들을 순회하면서 리사이즈 및 리스케일링
        for filename in os.listdir(input_dir):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                image_path = os.path.join(input_dir, filename)
                try:
                    # 이미지 읽기
                    img = load_img(image_path, target_size=(img_width, img_height))
                    x = img_to_array(img)  # 이미지를 numpy 배열로 변환
                    x = np.expand_dims(x, axis=0)  # 차원을 추가하여 배치 크기 형태로 변환

                    # 이미지 리스케일링
                    rescaled_image = next(datagen.flow(x, batch_size=1))[0]  # 배치 크기가 1인 리스케일된 이미지 추출
                    
                    # 리스케일된 이미지를 저장
                    rescaled_image_path = os.path.join(output_dir, filename)
                    rescaled_image = np.uint8(rescaled_image * 255)  # 원래의 픽셀 값 범위로 변환 (0-255)
                    save_img(rescaled_image_path, rescaled_image)
                
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

print("이미지 리사이즈 및 리스케일링이 완료되었습니다.")
