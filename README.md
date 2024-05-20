# Animal-Image-Preprocessing

### ✅ face_cut_folder.py

### 입력 디렉토리 경로 <br>
얼굴만 따올 이미지가 있는 폴더 경로 입력<br>
input_path = './image/여자 고양이상'

### 출력 디렉토리 경로 <br>
결과 폴더 경로 <br>
output_path = './1.detected_face'

최종 결과물은 1.detected_face 폴더에 저장됨


---
### ✅ face_resize_rescale.py

### 입력 디렉토리 경로
사진크기를 조절하고 리스케일을 적용할 이미지가 있는 폴더 경로 입력 <br>
input_path = './1.detected_face'

### 출력 디렉토리 경로
결과 폴더 <br>
output_path = './2.resize_rescaled_images'

---
### ✅ data_augmentation.py

### 입력 디렉토리 경로
input_path = './2.resize_rescaled_images'

### 출력 디렉토리 경로
output_path = './3.augmented_images'
