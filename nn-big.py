import numpy as np

# Aktivierungsfunktionen

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Standardisierung
def standard_scale(data, mean_val=None, std_val=None):
    if mean_val is None: mean_val = np.mean(data, axis=0)
    if std_val is None: std_val = np.std(data, axis=0) + 1e-8
    return (data - mean_val) / std_val, mean_val, std_val

# Datensatz erstellen mit linearen & nicht-linearen Sequenzen
# Lineare Sequenzen mit Addition und Subtraktion
dataset = np.array([[a + i for i in range(4)] for a in range(-200, 200)])

# Multiplikationssequenzen von a * 1 bis a * 20
multiplication_sequences = np.vstack([
    np.array([[a * i for i in range(1, 5)] for a in range(-200, 200)]) for i in range(1, 21)
])

# Additionssequenzen von a + 1 bis a + 10
addition_sequences = np.vstack([
    np.array([[a + i for i in range(j, j + 4)] for a in range(-200, 200)]) for j in range(1, 11)
])

# Subtraktionssequenzen von a - 1 bis a - 10
subtraction_sequences = np.vstack([
    np.array([[a - i for i in range(j, j + 4)] for a in range(-200, 200)]) for j in range(1, 11)
])

# Alles zusammenführen
final_dataset = np.vstack([dataset, multiplication_sequences, addition_sequences, subtraction_sequences])

dataset = np.vstack((dataset, final_dataset))
np.random.shuffle(dataset)

# Trainings- und Testdaten aufteilen
train_size = int(len(dataset) * 0.85)
train_data, test_data = dataset[:train_size], dataset[train_size:]
inputs_train, outputs_train = train_data[:, :-1], train_data[:, -1:]
inputs_test, outputs_test = test_data[:, :-1], test_data[:, -1:]

# Normalisierung
i_train, mean_i, std_i = standard_scale(inputs_train)
o_train, mean_o, std_o = standard_scale(outputs_train)
i_test = (inputs_test - mean_i) / std_i
o_test = (outputs_test - mean_o) / std_o

# Netzwerkarchitektur
input_size, hidden_size, output_size = 3, 256, 1
np.random.seed(42)
weights = [np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)]
weights.append(np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / hidden_size))
weights.append(np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size))

# Adam-Optimierung
learning_rate, beta1, beta2, epsilon = 0.01, 0.9, 0.999, 1e-8
m_weights = [np.zeros_like(w) for w in weights]
v_weights = [np.zeros_like(w) for w in weights]
batch_size, epochs = 1000, 20000
decay_factor, patience = 0.995, 500
best_loss, no_improvement_epochs = float("inf"), 0
loss_history = []

# Cosine Annealing Lernrate
def cosine_annealing(epoch):
    return 0.01 * (0.5 * (1 + np.cos(epoch / epochs * np.pi)))

# Training
for epoch in range(epochs):
    batch_size = min(batch_size, len(i_train))  # Sicherstellen, dass batch_size nie zu groß ist
    idx = np.random.choice(len(i_train), batch_size, replace=False)
    batch_inputs, batch_outputs = i_train[idx], o_train[idx]

    # Forward Pass
    a1 = leaky_relu(np.dot(batch_inputs, weights[0]))
    a2 = leaky_relu(np.dot(a1, weights[1]))
    final_output = np.dot(a2, weights[2])

    # Fehlerberechnung
    error = batch_outputs - final_output
    loss = np.mean(error**2)
    loss_history.append(loss)

    if loss < best_loss:
        best_loss, no_improvement_epochs = loss, 0
    else:
        no_improvement_epochs += 1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Verlust: {loss:.6f}, Lernrate: {learning_rate:.6f}")

    # Backpropagation
    d_output = error
    d_hidden2 = d_output.dot(weights[2].T) * leaky_relu_derivative(a2)
    d_hidden1 = d_hidden2.dot(weights[1].T) * leaky_relu_derivative(a1)

    gradients = [batch_inputs.T.dot(d_hidden1), a1.T.dot(d_hidden2), a2.T.dot(d_output)]

    # Adam-Update
    for i in range(len(weights)):
        m_weights[i] = beta1 * m_weights[i] + (1 - beta1) * gradients[i]
        v_weights[i] = beta2 * v_weights[i] + (1 - beta2) * (gradients[i] ** 2)
        m_hat = m_weights[i] / (1 - beta1 ** (epoch + 1))
        v_hat = v_weights[i] / (1 - beta2 ** (epoch + 1))
        weights[i] += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    learning_rate = cosine_annealing(epoch)

# Vorhersagefunktion
def predict(input_data):
    input_data = (input_data - mean_i) / std_i
    a1 = leaky_relu(np.dot(input_data, weights[0]))
    a2 = leaky_relu(np.dot(a1, weights[1]))
    final_output = np.dot(a2, weights[2])
    return final_output * std_o + mean_o

# Benutzerinteraktion
while True:
    try:
        user_input = np.array([float(input(f"Zahl {i+1}: ")) for i in range(3)]).reshape(1, -1)
        prediction = predict(user_input)
        print(f"[Neuronales Netzwerk] Vorhergesagte vierte Zahl: {prediction[0, 0]:.2f}")
        print("[SYSTEM] Bedenke, dass das Netzwerk darauf trainiert wurde, auch auf etwas kompliziertere Muster zu reagieren, und daher meist nur nahe liegt und nicht das perfekte Ergebniss liefet.")
    except ValueError:
        print("[SYSTEM] Ungültige Eingabe. Bitte Zahlen eingeben.")