import math
import random
import csv

# 파일 생성 및 헤더 작성
with open('time_value2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time", "value"])

    # 시간 및 value 값 생성
    for repeat in range(1):
        for hour in range(24):
            minute = 0
            # 시간 형식 설정
            time = f"{hour:02d}:{minute:02d}"
            # sin 함수를 이용해 value 생성 후 1~100 사이의 값으로 변환
            base_value = (math.sin(hour) * 0.5 + 0.5) * 99  # sin 결과를 0~99 사이의 값으로 변환
            random_addition = random.randint(0, 10)  # 0부터 10 사이의 랜덤한 수
            value = int(base_value + random_addition)
            value = max(1, min(100, value)) 
            writer.writerow([time, value])
print('생성')