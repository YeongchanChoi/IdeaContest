import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 데이터 프레임 로드 (time_value.csv에서 로드)
df = pd.read_csv('time_value.csv')

# 값만 사용
values = df['value'].values.astype(float)

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(-1, 1))
values_scaled = scaler.fit_transform(values.reshape(-1, 1))

# PyTorch 텐서로 변환
values_tensor = torch.FloatTensor(values_scaled).view(-1)

# 시퀀스 데이터 생성 함수
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 하이퍼파라미터 설정
seq_length = 5  # 사용할 과거 시점의 수
train_inout_seq = create_inout_sequences(values_tensor, seq_length)

# LSTM 모델 정의
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# 모델, 손실 함수, 최적화 알고리즘 설정
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
epochs = 150

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 5 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {epochs:3} loss: {single_loss.item():10.10f}')
# 모델 저장 경로 설정
MODEL_PATH = 'model.pth'

# 모델의 state_dict 저장
torch.save(model.state_dict(), MODEL_PATH)
######################################################################################################
# # 예측
# last_seq = values_tensor[-seq_length:]
# with torch.no_grad():
#     model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
#                          torch.zeros(1, 1, model.hidden_layer_size))
#     predicted_values = []
#     for _ in range(24):  # 다음 24시간을 예측
#         last_seq = last_seq.view(-1)
#         pred = model(last_seq)
#         predicted_values.append(pred.item())
#         # 새로운 예측값을 마지막 시퀀스에 추가하고 가장 오래된 값을 제거
#         last_seq = torch.cat((last_seq[1:], pred))

# # 예측된 값들을 원래 스케일로 변환
# predicted_values = np.array(predicted_values).reshape(-1, 1)
# predicted_values = scaler.inverse_transform(predicted_values).reshape(-1)

# print("다음 24시간 동안의 예상 값:")
# for i, value in enumerate(predicted_values, 1):
#     print(f"{i}시간 후: {value:.2f}")


#     import matplotlib.pyplot as plt

# # time_value2.csv에서 실제 값 로드
# df_test = pd.read_csv('time_value2.csv')
# test_values = df_test['value'].values.astype(float)

# # 데이터 스케일링 (학습 데이터에 사용된 scaler 활용)
# test_values_scaled = scaler.transform(test_values.reshape(-1, 1))

# # PyTorch 텐서로 변환
# test_values_tensor = torch.FloatTensor(test_values_scaled).view(-1)

# # 실제 값에 대한 시퀀스 데이터 생성
# test_inout_seq = create_inout_sequences(test_values_tensor, seq_length)

####################################################################################################
# # 예측 값 생성
# model.eval() # 평가 모드로 전환
# test_predictions = []
# with torch.no_grad():
#     for seq, _ in test_inout_seq:
#         model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
#                              torch.zeros(1, 1, model.hidden_layer_size))
#         test_predictions.append(model(seq).item())

# # 예측 값 스케일 되돌리기
# test_predictions_rescaled = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))

# # 실제 값과 예측 값 비교를 위한 Plot
# plt.figure(figsize=(10,6))
# plt.title('value vs predict')
# plt.xlabel('time')
# plt.ylabel('value')
# plt.plot(df_test['value'].values[seq_length:], label='value', color='blue') # 첫 seq_length 개의 값은 예측하지 않음
# plt.plot(test_predictions_rescaled, label='predict', color='red')
# plt.legend()
# plt.show()