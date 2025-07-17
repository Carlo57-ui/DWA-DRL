
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from AC_models import Actor, Critica
import torch.nn.functional as F
from Buffer import ReplayBuffer


# Parameters
num_episodes = 2
max_number_of_steps = 30
gamma = 0.9                   # Discount factor
learning_rate = 0.001         # Learning rate
tau = 0.1                     # Smoothing factor
policy_delay = 2              # Delay in policy update
batch_size = 1
buffer_size = 10000
dis_t = 0.1                    # Discount time of reward
reward = 0
ep_rew = 0
policy_noise = 0.1
noise_clip = 0.3
policy_freq = 2

    
input_size = 5
output_size = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Networks
actor = Actor(input_size, output_size).to(device)
critic1 = Critica().to(device)
critic2 = Critica().to(device)


# Optimizers
opt_actor = optim.Adam(actor.parameters(), lr=learning_rate)
opt_critico1 = optim.Adam(critic1.parameters(), lr=learning_rate)
opt_critico2 = optim.Adam(critic2.parameters(), lr=learning_rate)

# Load the model if it exists
try:
  actor.load_state_dict(torch.load('weights.pth', weights_only=True))
  print("Model loaded successfully.")
except FileNotFoundError:   
  print("Model not found.")
  
# Load the buffer if it exists
try:
  Buff = ReplayBuffer(buffer_size).loads()
  print("Buffer successfully.")
except FileNotFoundError:   
  Buff = ReplayBuffer(buffer_size)
  print("Buffer not found.")



terminated = False
inicio = time.time()

class Actor_Critic:
    def __init__(self, delta_x, delta_y, teta, v, w, obs):
        self.dx_t = torch.Tensor([delta_x]).unsqueeze(1)
        self.dy_t = torch.Tensor([delta_y]).unsqueeze(1)
        self.teta_t = torch.Tensor([teta]).unsqueeze(1)
        self.v_t = torch.Tensor([v]).unsqueeze(1)
        self.w_t = torch.Tensor([w]).unsqueeze(1)
        self.obs_t = torch.Tensor([obs]).unsqueeze(1)
        
        self.dist_objac = 25
        self.dist_objan = 25
    
    def actor(self):
        x = torch.cat((self.dx_t, self.dy_t, self.teta_t, self.v_t, self.w_t), dim=1)
        ap = actor.forward(x)  
        ap_n = ap * policy_noise
        self.ap = ap_n
        return ap      #original es ap_n
        
    def critic(self): 
        y = torch.cat((self.dx_t, self.dy_t, self.teta_t, self.v_t, self.w_t,self.ap), dim=1)
        q1 = critic1.forward(y)
        q2 = critic2.forward(y)
            
        q1.requires_grad_(True)  # Habilita el cálculo de gradiente para val_qp
        q2.requires_grad_(True)
            
        loss = F.mse_loss(q1, q2)
            
        opt_critico1.zero_grad()
        opt_critico2.zero_grad()
        loss.backward()
        opt_critico1.step()
        opt_critico2.step()
            
        #print("actualizado red")
        for param, param_pred in zip(critic2.parameters(), critic2.parameters()):
             param.data.copy_(tau * param_pred.data + (1 - tau) * param.data)
        for param, param_pred in zip(critic1.parameters(), critic1.parameters()):
             param.data.copy_(tau * param_pred.data + (1 - tau) * param.data)
        for param, param_pred in zip(actor.parameters(), actor.parameters()):
             param.data.copy_(tau * param_pred.data + (1 - tau) * param.data)
            
        torch.save(actor.state_dict(), 'weights.pth')
         
    def reward(self, dist_objf, dist_obji):
        # Si la distancia actual con el objetivo es menor que 1, la recompensa es 10
        if dist_obji < 1:
             r = 10
        # Si me acerco al objetivo la recompensa es 2
        elif dist_obji > dist_objf:
             r = 2
        else:
             r = -1
             
        return r
            
            
    def buff(self, state, action, n_state, reward):
        Buff.append((state, action, n_state, reward))
        Buff.save()
        

    
