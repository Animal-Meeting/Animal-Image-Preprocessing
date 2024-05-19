import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

# 입력 디렉토리 경로
input_path = './rescaled_images'

# 출력 디렉토리 경로
output_path = './augmented_images'

# 이미지 크기 설정 (예: 256x256)
img_width, img_height = 256, 256

# ImageDataGenerator를 사용하여 이미지 리스케일링 및 증강 설정
datagen = ImageDataGenerator(
    # rescale=1.0/255.0,           # 0-1 사이 값으로 리스케일링
    rotation_range=20,           # 랜덤 회전 각도
    # width_shift_range=0.2,       # 가로로 랜덤 이동
    # height_shift_range=0.2,      # 세로로 랜덤 이동
    shear_range=0.2,             # 랜덤 시어 변환
    zoom_range=0.2,              # 랜덤 줌
    horizontal_flip=True,        # 랜덤 가로 뒤집기
    brightness_range=[0.8, 1.2], # 랜덤 밝기 조절
    fill_mode='nearest'    
)

# 출력 디렉토리가 없으면 생성
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 입력 디렉토리의 이미지 파일들을 순회하면서 리사이즈, 리스케일링 및 증강
for folder in os.listdir(input_path):
    input_dir = os.path.join(input_path, folder)
    
    # 디렉토리인지 확인
    if os.path.isdir(input_dir):
        print("Start : " + input_dir)
        
        # 출력 디렉토리 경로 생성
        output_dir = os.path.join(output_path, folder)
        
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 입력 디렉토리의 이미지 파일들을 순회하면서 리사이즈, 리스케일링 및 증강
        for filename in os.listdir(input_dir):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                image_path = os.path.join(input_dir, filename)
                
                try:
                    # 이미지 읽기
                    img = load_img(image_path, target_size=(img_width, img_height))
                    x = img_to_array(img)  # 이미지를 numpy 배열로 변환
                    x = np.expand_dims(x, axis=0)  # 차원을 추가하여 배치 크기 형태로 변환

                    # 이미지 증강 및 저장
                    i = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=filename, save_format='jpg'):
                        i += 1
                        if i > 5:  # 이미지당 5개의 증강된 샘플 생성
                            break
                
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

print("이미지 리사이즈, 리스케일링 및 증강이 완료되었습니다.")
