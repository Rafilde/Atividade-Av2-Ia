import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.loadtxt('aerogerador.dat', delimiter='\t')

speed = data[:,0].reshape(-1,1) # Velocidade do vento
power = data[:,1].reshape(-1,1) # Potência do aerogerador

# Esse meu gráfico é não-linear
plt.scatter(speed, power, color='blue', label='Dados Normalizados')
plt.title('Velocidade do Vento vs Potência (Normalizado)')
plt.xlabel('Velocidade Normalizada')
plt.ylabel('Potência Normalizada')
plt.legend()
plt.show() 

# MODELO ANALINE-------------------------------------------------

# NORMALIZANDO OS DADOS ----------

speed_min = speed.min()
speed_max = speed.max()
speed_norm = (speed - speed_min) / (speed_max - speed_min)

power_min = power.min()
power_max = power.max()
power_norm = (power - power_min) / (power_max - power_min)

# ---------------------------------

# Fazendo a divisão 80% treino, 20% teste ----------
indices = np.random.permutation(len(speed_norm))
train_size = int(0.8 * len(speed_norm))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train = speed_norm[train_idx]
y_train = power_norm[train_idx]
X_test = speed_norm[test_idx]
y_test = power_norm[test_idx]

# ---------------------------------

# Papai está aplicando o BIAS (-1) ----------

X_train = np.hstack([np.full((X_train.shape[0], 1), -1), X_train])  
X_test = np.hstack([np.full((X_test.shape[0], 1), -1), X_test])

# ---------------------------------
