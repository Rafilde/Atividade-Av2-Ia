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

# ------------------------------------------------------------------------------------

# NORMALIZANDO OS DADOS em escala ----------

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

# MODELO ANALINE -------------------------------------------------

# ALGORITMO ADALINE (Treinamento) ----------

def train_adaline(X, d, eta, max_epochs, epsilon):
    N, n = X.shape  
    w = np.random.uniform(-0.1, 0.1, n)  
    epoch = 0
    EQM_previous = 0

    while True:

        u = X @ w  
        y = u  
        EQM = np.sum((d - u) ** 2) / N  
        
        
        if epoch > 0 and abs(EQM - EQM_previous) <= epsilon or epoch >= max_epochs:
            break

        
        for t in range(N):
            u_t = X[t] @ w  
            w += eta * (d[t] - u_t) * X[t]  

        EQM_previous = EQM
        epoch += 1

    return w, epoch, EQM

# ---------------------------------

# ALGORITMO ADALINE (Teste) ----------

def test_adaline(X, w):
    u = X @ w  
    y = u  
    return y

# ---------------------------------

# Hiperparâmetros que escolho ----------

eta = 0.01  # Taxa de aprendizado
max_epochs = 1000  # Número máximo de épocas
epsilon = 1e-5  # Precisão para early stopping

# ---------------------------------
# Treinar o modelo ----------

w, epochs, EQM_final = train_adaline(X_train, y_train, eta, max_epochs, epsilon)
print(f"Treinamento concluído em {epochs} épocas. EQM final: {EQM_final:.6f}")
print(f"Pesos finais: {w}")

# ---------------------------------
# Testar o modelo ----------

y_pred = test_adaline(X_test, w)

# Calcular EQM no conjunto de teste ----------

EQM_test = np.mean((y_test - y_pred) ** 2)
print(f"EQM no conjunto de teste (ADALINE): {EQM_test:.6f}")

# ---------------------------------

# MODELO MLP -------------------------------------------------

# Função de ativação sigmoide ----------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ---------------------------------
# Derivada da sigmoide ----------

def sigmoid_derivative(x):
    return x * (1 - x)

# ---------------------------------

# Configurando a topologia da rede MLP ----------

L = 1  # Número de camadas ocultas
q = [2]  # Número de neurônios em cada camada oculta (q1 = 2)
eta_mlp = 0.01  # Taxa de aprendizado
max_epochs_mlp = 1000  # Número máximo de épocas

# Dimensões
n_inputs = X_train.shape[1]  
n_outputs = 1  

# Criar listas para pesos W, saídas y, somas ponderadas v, e deltas
W = [None] * (L + 1)  
y = [None] * (L + 2) 
v = [None] * (L + 1)  

# Inicializar pesos (valores entre -0.5 e 0.5)
W[0] = np.random.uniform(-0.5, 0.5, (n_inputs, q[0]))  
W[1] = np.random.uniform(-0.5, 0.5, (q[0], n_outputs)) 

EQM = -1
epoch = 0

# ---------------------------------

# ALGORITMO MLP ----------

while epoch < max_epochs_mlp:
    
    EQM_old = EQM
    EQM = 0
    for i in range(len(X_train)):
        X_sample = X_train[i].reshape(-1, 1)
        d = y_train[i].reshape(-1, 1)

        
        y[0] = X_sample  
        for l in range(L + 1):
            v[l] = W[l].T @ y[l]  
            y[l + 1] = sigmoid(v[l])  

        
        EQM += np.sum((d - y[L + 1]) ** 2)

    EQM = EQM / (2 * len(X_train))

    
    for i in range(len(X_train)):
        X_sample = X_train[i].reshape(-1, 1)
        d = y_train[i].reshape(-1, 1)


        y[0] = X_sample
        for l in range(L + 1):
            v[l] = W[l].T @ y[l]
            y[l + 1] = sigmoid(v[l])

         
        delta = [None] * (L + 2)
        delta[L + 1] = (d - y[L + 1]) * sigmoid_derivative(y[L + 1])  
        for j in range(L - 1, -1, -1): 
            delta[j + 1] = (W[j + 1] @ delta[j + 2]) * sigmoid_derivative(y[j + 1])  

        
        for l in range(L + 1):
            W[l] += eta_mlp * (y[l] @ delta[l + 1].T)

    epoch += 1

y_pred_mlp = np.zeros(len(X_test))
for i in range(len(X_test)):
    X_sample = X_test[i].reshape(-1, 1)
    y[0] = X_sample
    for l in range(L + 1):
        v[l] = W[l].T @ y[l]
        y[l + 1] = sigmoid(v[l])
    y_pred_mlp[i] = y[L + 1].item()

# ---------------------------------

# Calcular EQM no conjunto de teste ----------

EQM_test_mlp = np.mean((y_test.flatten() - y_pred_mlp) ** 2)
print(f"EQM no conjunto de teste (MLP): {EQM_test_mlp:.6f}")

# ---------------------------------