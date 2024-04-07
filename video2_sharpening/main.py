import videocapture
import os
import urllib3
import json
import base64
import csv
import cv2 # OpenCV 라이브러리 추가

def main():
    video_path = 'video.mp4'
    times = [i * 0.066 for i in range(21)]
    print(times)
    output_folder = './pictures/'
    videocapture.capture_and_save_frames(video_path, times, output_folder)
    
    pictures_folder = "pictures"
    openApiURL = "http://aiopen.etri.re.kr:8000/ObjectDetect"
    accessKey = "ba46df83-1a80-41c9-b5f5-e6e749dfc316"
    
    with open('timetable.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Object Count"])
        
        for filename in os.listdir(pictures_folder):
            if filename.endswith(".jpg"):
                print(filename, " 작업중..")
                imageFilePath = os.path.join(pictures_folder, filename)
                
                # 이미지 불러오기
                original_image = cv2.imread(imageFilePath)
                
                # 이미지 타입을 float64로 변환
                original_image_float = original_image.astype('float64')
                
                # 라플라시안 계산
                laplacian = cv2.Laplacian(original_image_float, cv2.CV_64F)
                
                # 라플라시안 결과를 원본 이미지에 더하기
                sharpened_image_float = cv2.add(original_image_float, laplacian)
                
                # 결과 이미지를 uint8로 변환
                sharpened_image = cv2.convertScaleAbs(sharpened_image_float)

                
                # 선명해진 이미지를 임시 경로에 저장
                sharpened_image_path = os.path.join(pictures_folder, f"sharpened_{filename}")
                cv2.imwrite(sharpened_image_path, sharpened_image)
                
                # 선명해진 이미지로 객체 인식 작업 수행
                type = "jpg"
                with open(sharpened_image_path, "rb") as imageFile:
                    imageContents = base64.b64encode(imageFile.read()).decode("utf8")
                requestJson = {
                    "argument": {
                        "type": type,
                        "file": imageContents
                    }
                }
                http = urllib3.PoolManager()
                response = http.request(
                    "POST",
                    openApiURL,
                    headers={"Content-Type": "application/json; charset=UTF-8", "Authorization": accessKey},
                    body=json.dumps(requestJson)
                )
                if response.status == 200:
                    response_data = json.loads(response.data.decode('utf-8'))
                    objects = response_data.get('return_object', {}).get('data', [])
                    object_count = len(objects)
                    print(filename, object_count)
                    filename2 = filename[0:11]
                    writer.writerow([filename2, object_count])
                else:
                    print(f"Error: {filename} could not be processed. Response code: {response.status}")
                # 선명해진 이미지 파일 삭제
                os.remove(sharpened_image_path)

if __name__ == "__main__":
    main()
