import torch
import chess
import numpy as np
import random
import methods as m

hidden_size = 128; num_classes = 499
embed_dim = 128
input_size = 98
num_layers = 1

model2 = m.myLSTM(input_size, hidden_size, num_layers, num_classes, embed_dim)
model2.load_state_dict(torch.load('Deeplefish.pt'))
model2.eval()

while True:
  msg = input()
  if msg == "uci":
    print("Deeplefish by Rafi Indra Fattah")
    print("uciok")

  elif msg == "isready":
    print("readyok")

  elif msg==("ucinewgame"):
   continue

  elif msg.startswith("position startpos"):
    m.board.reset()
    if len(msg.split()) <=3:
      choice = ["e4", "d4"]
      move = choice[random.randint(0,1)]
      continue
    moves = msg.split(" ")[3:]
    moves2 = []
    for move in moves:
      moves2.append(m.board.san(m.board.parse_san(move)))
      m.board.push_san(move)
    data = m.encode(moves2)
    data = torch.tensor([[data]])
    print(data)
    nextMove = model2(data)
    sorted = np.argsort(nextMove[0].detach().numpy())[::-1]
    prediksi = np.argmax(nextMove.detach())
    move = m.dictionary[prediksi.item()]
    print(move)

    move = m.postprocess(nextMove)[0]

  elif msg.startswith("go"):
   print(f"bestmove {m.board.push_san(move)}")
  elif msg  == "quit":
    break