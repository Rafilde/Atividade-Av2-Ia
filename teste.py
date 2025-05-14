import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Funções fornecidas (já no seu código, incluídas para completude)
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

def test_adaline(X, w):
    u = X @ w
    y = u
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_mlp(X_train, y_train, X_test, y_test, L, q, eta, max_epochs):
    n_inputs = X_train.shape[1]
    n_outputs = 1
    W = [None] * (L + 1)
    y = [None] * (L + 2)
    v = [None] * (L + 1)
    
    W[0] = np.random.uniform(-0.5, 0.5, (n_inputs, q[0]))
    for l in range(1, L):
        W[l] = np.random.uniform(-0.5, 0.5, (q[l-1], q[l]))
    W[L] = np.random.uniform(-0.5, 0.5, (q[-1], n_outputs))
    
    for epoch in range(max_epochs):
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
                W[l] += eta * (y[l] @ delta[l + 1].T)
    
    y_pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        X_sample = X_test[i].reshape(-1, 1)
        y[0] = X_sample
        for l in range(L + 1):
            v[l] = W[l].T @ y[l]
            y[l + 1] = sigmoid(v[l])
        y_pred[i] = y[L + 1].item()
    
    EQM_test = np.mean((y_test.flatten() - y_pred) ** 2)
    return EQM_test

# Carregar e normalizar os dados (já no seu código)
data = np.loadtxt('aerogerador.dat', delimiter='\t')
speed = data[:,0].reshape(-1,1)
power = data[:,1].reshape(-1,1)

speed_min = speed.min()
speed_max = speed.max()
speed_norm = (speed - speed_min) / (speed_max - speed_min)

power_min = power.min()
power_max = power.max()
power_norm = (power - power_min) / (power_max - power_min)

# Hiperparâmetros
eta_adaline = 0.01
max_epochs_adaline = 1000
epsilon_adaline = 1e-5

eta_mlp = 0.01
max_epochs_mlp = 1000
L_mlp = 1
q_mlp = [2]

# Monte Carlo Validation
R = 250
mse_adaline = np.zeros(R)
mse_mlp = np.zeros(R)
n_samples = len(speed_norm)

for r in range(R):
    # Particionamento aleatório
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train = speed_norm[train_idx]
    y_train = power_norm[train_idx]
    X_test = speed_norm[test_idx]
    y_test = power_norm[test_idx]
    
    # Adicionar bias
    X_train_bias = np.hstack([np.full((X_train.shape[0], 1), -1), X_train])
    X_test_bias = np.hstack([np.full((X_test.shape[0], 1), -1), X_test])
    
    # Treinar e testar ADALINE
    w, _, _ = train_adaline(X_train_bias, y_train, eta_adaline, max_epochs_adaline, epsilon_adaline)
    y_pred_adaline = test_adaline(X_test_bias, w)
    mse_adaline[r] = np.mean((y_test - y_pred_adaline) ** 2)
    
    # Treinar e testar MLP
    mse_mlp[r] = train_mlp(X_train_bias, y_train, X_test_bias, y_test, L_mlp, q_mlp, eta_mlp, max_epochs_mlp)

# Calcular estatísticas
stats = {
    'ADALINE': {
        'Média': np.mean(mse_adaline),
        'Desvio-Padrão': np.std(mse_adaline),
        'Maior Valor': np.max(mse_adaline),
        'Menor Valor': np.min(mse_adaline)
    },
    'MLP': {
        'Média': np.mean(mse_mlp),
        'Desvio-Padrão': np.std(mse_mlp),
        'Maior Valor': np.max(mse_mlp),
        'Menor Valor': np.min(mse_mlp)
    }
}

# Exibir tabela
print("\nResultados da Validação Monte Carlo (R=250):")
print(f"{'Modelos':<30} {'Média':<12} {'Desvio-Padrão':<15} {'Maior Valor':<12} {'Menor Valor':<12}")
print("-" * 80)
for model, metrics in stats.items():
    print(f"{model:<30} {metrics['Média']:<12.6f} {metrics['Desvio-Padrão']:<15.6f} {metrics['Maior Valor']:<12.6f} {metrics['Menor Valor']:<12.6f}")

# Opcional: Visualizar resultados com boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([mse_adaline, mse_mlp], labels=['ADALINE', 'MLP'])
plt.title('Distribuição do MSE (Monte Carlo, R=250)')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig('mse_boxplot.png')
plt.show()