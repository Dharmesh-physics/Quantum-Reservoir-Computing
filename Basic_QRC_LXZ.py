import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import special
from scipy import linalg
from qutip import Qobj
import scipy.stats as stats
from scipy.integrate import solve_ivp

ket0 = qt.basis(2, 0)  # |0>
ket1 = qt.basis(2, 1)  # |1>

def generate_hamiltonian(n, gamma, J):
    H = 0
    for i in range(n):
        H += gamma * qt.tensor([qt.sigmaz() if j == i else qt.qeye(2) for j in range(n)])
    for i in range(n):
        for j in range(i + 1, n):
            H += J[i, j] * qt.tensor([qt.sigmax() if k == i or k == j else qt.qeye(2) for k in range(n)])
    return H

def generate_gue_matrix(alpha,dimension):
    A = np.random.randn(dimension, dimension) + 1j * np.random.randn(dimension, dimension)
    GUE_matrix = (A + A.conj().T) / 2
    diagonal_matrix = np.diag(np.random.rand(dimension) + 1j * np.random.rand(dimension))
    h_matrix = (alpha * diagonal_matrix) + ((1-alpha) * GUE_matrix)
    return GUE_matrix

def reshape_gue_to_qubits(gue_qobj, n_qubits):
    matrix_data = gue_qobj.full()
    reshaped_matrix = matrix_data.reshape([2] * n_qubits * 2) 
    flattened_matrix = reshaped_matrix.reshape(2**n_qubits, 2**n_qubits)
    reshaped_qobj = Qobj(flattened_matrix, dims=[[2]*n_qubits, [2]*n_qubits])
    return reshaped_qobj

def partial_trace_over_first_qubit(rho_N, N):
    keep_indices = list(range(1, N))
    rho_partial = rho_N.ptrace(keep_indices)
    return rho_partial

def z_i_operator(N,i):
    op_list = [qt.qeye(2)] * N
    op_list[i] = qt.sigmaz()
    operator = qt.tensor(op_list)
    return operator

def input_encoding(classical_input,rho,N):
    s_k = classical_input
    p0 = 1 - s_k  # Probability for |0>
    p1 = s_k  # Probability for |1>
    rho_first_qubit = p0 * ket0 * ket0.dag() + p1 * ket1 * ket1.dag()  # density matrix for first qubit
    rho_first_qubit.dims = [[2], [2]]
    rho_partial = partial_trace_over_first_qubit(rho, N)
    rho_input = qt.tensor([rho_first_qubit, rho_partial])
    return rho_input

def QRC(Input, rho, H, N, total_time, transient, training, testing, V, tao):
    Input = (Input - np.min(Input)) / (np.max(Input) - np.min(Input))
    time = tao
    iota = complex(0, 1)
    unit_unitary = (H * time * iota).expm()
    V_unitary = (H * (tao / V) * iota).expm()
    training_output = np.zeros(((N*V) + 1, training))
    testing_output = np.zeros(((N*V) + 1, testing))
    for k in range(training):
        training_output[0, k] = 1
    for k in range(testing):
        testing_output[0, k] = 1
    for i in range(total_time):
        if i < transient:
            rho_input = input_encoding(Input[i],rho,N)
            rho = (unit_unitary.dag()) * rho_input * unit_unitary
        elif transient <= i < (training + transient):
            rho_input = input_encoding(Input[i],rho,N)
            for j in range(V):
                rho_output = (V_unitary.dag()) * rho_input * V_unitary
                # Output storage
                for k in range(N):
                    z_i = z_i_operator(N, k)
                    node = (z_i * rho_output).tr()
                    training_output[j*N + k + 1, (i-transient)] = (node.real + 1) / 2
                rho_input = rho_output 
            rho = rho_output 
        else:
            rho_input = input_encoding(Input[i],rho,N)
            for j in range(V):
                rho_output = (V_unitary.dag()) * rho_input * V_unitary
                # Output storage
                for k in range(N):
                    z_i = z_i_operator(N, k)
                    node = (z_i * rho_output).tr()
                    testing_output[j*N + k + 1, (i-(training+transient))] = (node.real + 1) / 2
                rho_input = rho_output
            rho = rho_output
    return training_output, testing_output

def capacity(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    cov = np.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
    C = cov / (np.std(y_true) * np.std(y_pred))
    return C

def generate_narma10_sequence(length):
    n = 7 # narma order
    alpha = 2.11
    beta = 3.73
    Gamma = 4.11
    T = 100
    Input = np.zeros(length)
    for i in range(total_time):
        Input[i] = 0.1 * (math.sin((2*math.pi*alpha*i)/T) * math.sin((2*math.pi*beta*i)/T) * math.sin((2*math.pi*Gamma*i)/T) + 1)
    u = Input
    y = np.zeros(length)  # Initialize the output sequence
    for t in range(n, length):
        y[t] = (0.3 * y[t-1]) + (0.05 * (y[t-1]) * np.sum(y[t-n:t])) + (1.5 * (u[t-n]) * (u[t])) + 0.1
    Input = (Input - np.min(Input)) / (np.max(Input)-np.min(Input))
    return Input,y

def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def time_delay_function(time,total_time):
    Input = np.random.uniform(0,1,total_time)
    Output = np.zeros(total_time)
    for i in range(time , total_time):
        Output[i] = Input[i-time]
    return Input,Output

ensemble_size = 10
capacities = np.zeros(ensemble_size)
mse_array = np.zeros(ensemble_size)

for ensemble_index in range(ensemble_size) :
    # ======= Reservoir Definition ======= #
    N = 8
    tao = 1
    V = 1
    '''
    gamma = 5
    J_0 = 10
    J = np.random.uniform( 0.9*J_0, 1.1*J_0, (N, N))
    J = (J + J.T) / 2
    H = generate_hamiltonian(N, gamma, J)
    '''
    gue_matrix = generate_gue_matrix(0,2**N)
    gue_qobj = Qobj(gue_matrix)
    H = reshape_gue_to_qubits(gue_qobj, N)

    # ======= Input and output sequence ======= #
    transient = 100
    training = 500
    testing = 200
    total_time = transient + training + testing

    # Lorenz XZ Sequence :-
    initial_state = [1.0, 1.0, 1.0]
    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], total_time)
    solution = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval)
    x = solution.y[0]
    Lorenz_Input = ( x - np.min(x) ) / ( np.max(x) - np.min(x) )
    Lorenz_Output = solution.y[2]

    # ======= Doing QRC ======= #
    rho = qt.rand_dm(2**N)
    rho.dims = [[2] * N, [2] * N]
    training_output, testing_output = QRC(Lorenz_Input, rho, H, N, total_time, transient, training, testing, V, tao)

    # Linear regression
    # Training - 
    Y_dependent = Lorenz_Output[transient:transient+training]
    X_independent = np.transpose(training_output)
    model = LinearRegression()
    model.fit(X_independent, Y_dependent)
    predictions = model.predict(X_independent)

    # Testing - 
    Y_dependent = Lorenz_Output[transient+training:transient+training+testing]
    X_independent = np.transpose(testing_output)
    predictions = model.predict(X_independent)
    capacities[ensemble_index] = capacity(Y_dependent,predictions) 
    mse_array[ensemble_index] = np.mean((Y_dependent - predictions) ** 2)
    print(f"Ensemble {ensemble_index+1}/{ensemble_size} - Capacity: {capacities[ensemble_index]:.4f}, MSE: {mse_array[ensemble_index]:.4f}")

print(f"Mean Capacity: {np.mean(capacities):.4f}, Standard Deviation Capacity: {np.std(capacities):.4f}")
print(f"Mean MSE: {np.mean(mse_array):.4f}, Standard Deviation MSE: {np.std(mse_array):.4f}")

