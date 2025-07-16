
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
  def __init__(self, input_size, output_size):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(input_size, 250)
    self.fc2 = nn.Linear(250, 200)
    self.fc4 = nn.Linear(200, output_size)    
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc4(x)
    x = self.softmax(x)
    return x

class Critica(nn.Module):
  def __init__(self):
    super(Critica, self).__init__()
    self.fc1 = nn.Linear(7, 250)
    self.fc2 = nn.Linear(250, 200)
    self.fc3 = nn.Linear(200, 1)
    self.relu = nn.ReLU()


  def forward(self, y): 
      Q = self.relu(self.fc1(y))
      Q = self.relu(self.fc2(Q))
      Q = F.sigmoid(self.fc3(Q))
      
      return Q

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)

#         self.mu_head = nn.Linear(hidden_dim, action_dim)
#         self.log_std_head = nn.Linear(hidden_dim, action_dim)

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         # Si el input no es secuencia, agregar dimensión para LSTM
#         if len(x.shape) == 2:
#             x = x.unsqueeze(1)  # (batch, seq=1, feature)
#         lstm_out, _ = self.lstm(x)
#         x = F.relu(self.fc3(lstm_out[:, -1, :]))  # última salida temporal

#         mu = self.mu_head(x)
#         log_std = self.log_std_head(x)
#         std = torch.exp(log_std)
#         return mu, std


# class Critica(nn.Module):
#     def __init__(self, hidden_dim=256):
#         super(Critica, self).__init__()
#         self.fc1 = nn.Linear(7, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         self.q_out = nn.Linear(hidden_dim, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         if len(x.shape) == 2:
#             x = x.unsqueeze(1)
#         lstm_out, _ = self.lstm(x)
#         x = F.relu(self.fc3(lstm_out[:, -1, :]))
#         q = self.q_out(x)
#         return q
