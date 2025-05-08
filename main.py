import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.loadtxt('aerogerador.dat', delimiter='\t')

speed = data[:,0].reshape(-1,1) # Velocidade do vento
power = data[:,1].reshape(-1,1) # Potência do aerogerador

# NORMALIZAÇÃO DOS DADOS ----------

speed_min = speed.min()
speed_max = speed.max()
speed_norm = (speed - speed_min) / (speed_max - speed_min)

power_min = power.min()
power_max = power.max()
power_norm = (power - power_min) / (power_max - power_min)

# ---------------------------------

plt.scatter(speed_norm, power_norm, color='blue', label='Dados Normalizados')
plt.title('Velocidade do Vento vs Potência (Normalizado)')
plt.xlabel('Velocidade Normalizada')
plt.ylabel('Potência Normalizada')
plt.legend()
plt.show() # é não linear
