import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('aerogerador.dat', delimiter='\t')

x = data[:,0].reshape(-1,1) # Velocidade do vento
y = data[:,1].reshape(-1,1) # Potência do aerogerador

# precisa normalizar ?

plt.scatter(x,y)
plt.scatter(x, y, color='blue', label='Dados')
plt.title('Gráfico de Dispersão: Velocidade do Vento e Potência Gerada')
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.legend()
plt.grid(True)
plt.show() # é não linear