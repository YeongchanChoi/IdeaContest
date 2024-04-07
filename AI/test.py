import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# LSTM 모델 정의 (학습 과정에서 사용한 모델과 동일한 구조여야 함)
class LSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size)

        self.linear = torch.nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# 모델 로드
model = LSTM()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 데이터 스케일링을 위한 MinMaxScaler 준비
scaler = MinMaxScaler(feature_range=(-1, 1))

# 시퀀스 데이터 생성 함수
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 원본 데이터 로드 및 스케일링
df = pd.read_csv('time_value.csv')
values = df['value'].values.astype(float)
scaler.fit(values.reshape(-1, 1))  # 학습 데이터에 사용한 스케일러 조정

# 예측 시작
seq_length = 5  # 과거 시점의 수
values_tensor = torch.FloatTensor(scaler.transform(values.reshape(-1, 1))).view(-1)

last_seq = values_tensor[-seq_length:]
with torch.no_grad():
    predicted_values = []
    for _ in range(24):  # 다음 24시간을 예측
        last_seq = last_seq.view(-1)
        pred = model(last_seq)
        predicted_values.append(pred.item())
        last_seq = torch.cat((last_seq[1:], pred))

# 예측된 값들을 원래 스케일로 변환
predicted_values = np.array(predicted_values).reshape(-1, 1)
predicted_values = scaler.inverse_transform(predicted_values).reshape(-1)

print("다음 24시간 동안의 예상 값:")
for i, value in enumerate(predicted_values, 1):
    print(f"{i}시간 후: {value:.2f}")

# time_value2.csv 파일 로드 및 스케일링
df2 = pd.read_csv('time_value2.csv')
actual_values = df2['value'].values.astype(float)
actual_values_scaled = scaler.transform(actual_values.reshape(-1, 1)).reshape(-1)

# 실제 값과 예측 값의 길이를 동일하게 조정
# 예측은 24시간 동안의 값을 출력하므로, 실제 값에서도 마지막 24개의 데이터만 사용
actual_values_scaled = actual_values_scaled[-24:]

# 실제 값과 예측 값 비교를 위한 plot 생성
plt.figure(figsize=(10, 6))
plt.plot(predicted_values, label='Predicted Values', color='red')
plt.plot(scaler.inverse_transform(actual_values_scaled.reshape(-1, 1)).reshape(-1), label='Actual Values', color='blue')
plt.title('Comparison between Actual and Predicted Values')
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.legend()
plt.show()
