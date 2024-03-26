import numpy as np
import re
import pandas as pd
import chess
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import chess.engine
import methods as m

init  = []
temp = ""
with open("RawData.txt", "r") as file:
  for line in file:
    if (line.split()):
      temp += line.replace("$2", "").replace("$4", "").replace("$6", "")
    else:
      init.append(temp.split()[:-1])
      temp = ""
file.close()

cleaned = [game for game in init if len(game) < 100]
print("cleaning data done")
remove_list = ['x', '+', '#', '=Q', '=R', '=N', '=B']
raw = copy.deepcopy(cleaned)
for i, game in enumerate(cleaned):
  for j, move in enumerate(game):
    for l in remove_list:
      move = move.replace(l, "")
    if len(move) > 3 and move[0] != "O":
      move = move[0] + move[-2] + move[-1]
    cleaned[i][j] = move

found = []
dictionary = {}
flippedDict = {}
for game in cleaned:
  for move in game:
    if move not in found:
      found.append(move)

for i, move in enumerate(found):
  dictionary[i+1] = move
  flippedDict[move] = i+1
print("building dictionary done")

for i, data in enumerate(cleaned):
    for j, move in enumerate(data):
        cleaned[i][j] = flippedDict[move]
print("encoding data done")

seqRaw = [raw[i][:2*j+3] for i, line in enumerate(cleaned) for j in range(len(line)//2)]
seq = [line[:2*j+3] for i, line in enumerate(cleaned) for j in range(len(line)//2)]
print("generating N-grams done")

for i, game in enumerate(seq):
  for j in range(99 - len(game)):
    seq[i].insert(0, 0)
print("padding data done")

label = [sequence.pop() for sequence in seq]
print("detaching labels done")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

split = len(seq) * 8 // 10
print(split)

featured = torch.Tensor(seq[:split]).to(device)
labeled = torch.Tensor(label[:split]).to(device)
dataset = TensorDataset(featured, labeled)
batch_size = 250
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, pin_memory=True, shuffle=True)

print("training model started....")
hidden_size = 128; num_classes = 499
embed_dim = 128
input_size = 98
num_layers = 1
num_epochs = 10
engine = chess.engine.SimpleEngine.popen_uci("C:\ChessStuff\pythonProject18\pythonProject\stockfish\stockfish-windows-x86-64-avx2.exe")
model = m.myLSTM(input_size, hidden_size, num_layers, num_classes, embed_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_total_steps = len(dataloader)
temp = float("inf")
for epoch in range(num_epochs):
  lossFinal = 0
  i = 0
  for images, labeltot in dataloader:
    images = images.reshape(1, batch_size, input_size).to(device)
    outputs = model(images)
    loss = criterion(outputs, labeltot.long())
    lossFinal += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    i += 1

    if (i + 1) % 10000 == 0:
      print(
        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Current Batch Loss: {loss.item():.4f}, Overall Train Loss: {lossFinal / (i + 1)}')
  name = str(epoch + 1) + "model.pt"
  torch.save(model.state_dict(), name)
  lossFinal = lossFinal / (i + 1)
  print(f"Loss Final {lossFinal}")
  if lossFinal > temp:
    print("done")
    break
  else:
    temp = lossFinal

  with torch.no_grad():
    criterion2 = nn.CrossEntropyLoss()
    n_correct = 0
    acl = 0
    j = 0
    illegal = 0
    losss = 0
    for rangkaian, next in zip(torch.tensor(seq[split:]), torch.tensor(label[split:])):
      rangkaian = rangkaian.reshape(1, 1, input_size).to(device)
      logitsRangkaian = model(rangkaian)
      _, prediksi = torch.max(logitsRangkaian.data, 1)
      max = float('-inf')
      m.board.reset()

      for kintil in seqRaw[split + j][:-1]:
        m.board.push_san(kintil)
      info = engine.analyse(m.board, chess.engine.Limit(depth=5))
      temp = info["score"].white().score(mate_score=1500)

      moveInit = dictionary[prediksi.item()]
      processed = m.postprocess(logitsRangkaian)
      move = processed[0]
      if processed[1]:
        illegal += 1

      m.board.push_san(move)
      info2 = engine.analyse(m.board, chess.engine.Limit(depth=5))
      temp2 = info2["score"].white().score(mate_score=1500)
      acl += (temp - temp2)
      j += 1
      if j % 10000 == 0:
        print(f'Row: {j}, Current ACPL: {temp - temp2}, Overall ACPL: {acl / j}')
      if move == dictionary[next.item()]:
        n_correct += 1
      losss += criterion2(logitsRangkaian, torch.tensor([next]).long())
    print(logitsRangkaian.size())
    print(next)
    losss /= j
    acc = 100.0 * n_correct / j
    print(f'Val Loss {losss}, Val Acc: {acc}, Val ACPL: {acl / j}, Illegal Move Count: {illegal}/{j}')
