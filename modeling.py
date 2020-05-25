from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute, IBMQ
from qiskit.compiler import transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.circuit import Gate

import numpy as np



def cos_classifier(train, test, label_train, printing=False):
    # x_train = train
    # x_new = test
    # y_train = label_train
    c = ClassicalRegister(1, 'c')
    x_train = QuantumRegister(1, 'x^{(b)}')
    x_test = QuantumRegister(1, 'x^{(test)}')
    y_train = QuantumRegister(1, 'y^{(b)}')
    y_test = QuantumRegister(1, 'y^{(test)}')
    qc = QuantumCircuit(x_train, x_test, y_train, y_test, c)
    qc.initialize(train, [x_train[0]])
    qc.initialize(test, [x_test[0]])
    qc.initialize(label_train, [y_train[0]])
    qc.barrier()
    qc.h(y_test)
    qc.cswap(y_test, x_train, x_test)
    qc.h(y_test)
    qc.barrier()
    qc.cx(y_train, y_test)
    qc.measure(y_test, c)
    if printing:
        print(qc)
    return qc






def ensemble(X_data, Y_data, x_test, n_swap=1, d=2, balanced=True):
    # d = 2  # number of control qubits
    # n_swap = 1
    # balanced = True

    n_obs = len(X_data)
    # if n_obs != len(Y_data):
    #     return print('Error: in the input size')

    n_reg = d + 2 * n_obs + 1  # total number of registers

    control = QuantumRegister(d)
    data = QuantumRegister(n_obs, 'x')
    labels = QuantumRegister(n_obs, 'y')
    data_test = QuantumRegister(1, 'test_data')
    label_test = QuantumRegister(1, 'test_label')
    c = ClassicalRegister(1)


    qc = QuantumCircuit(control, data, labels, data_test, label_test, c)

    qc.initialize(x_test, [data_test[0]])

    for index in range(n_obs):
        qc.initialize(X_data[index], [data[index]])
        qc.initialize(Y_data[index], [labels[index]])

    for i in range(d):
        qc.h(control[i])

    if balanced:
        for i in range(d-1):
            for j in range(n_swap):
                U = np.random.choice(range(int(n_obs / 2)), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                U = np.random.choice(range(int(n_obs / 2), n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

            qc.x(control[i])

            for j in range(n_swap):
                U = np.random.choice(range(int(n_obs / 2)), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                U = np.random.choice(range(int(n_obs / 2), n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                qc.barrier()
        U = np.random.choice(range(int(n_obs / 2)), 1, replace=False)
        U = np.insert(U, 1, n_obs - 1)
        qc.cswap(control[d-1], data[int(U[0])], data[int(U[1])])
        qc.cswap(control[d-1], labels[int(U[0])], labels[int(U[1])])

        qc.x(control[d-1])
    else:
        for i in range(d):
            for j in range(n_swap):
                U = np.random.choice(range(n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

            qc.x(control[i])

            for j in range(n_swap):
                U = np.random.choice(range(n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

    qc.barrier()
    # C
    ix_cls = n_obs - 1
    qc.h(label_test[0])
    qc.cswap(label_test[0], data[ix_cls], data_test[0])
    qc.h(label_test[0])
    qc.cx(labels[ix_cls], label_test[0])
    qc.measure(label_test[0], c)
    return qc



def ensemble_fixed_U(X_data, Y_data, x_test, d = 2 ):
    #d = 2  # number of control qubits
    n_obs = len(X_data)

    if n_obs != len(Y_data):
        return print('Error: in the input size')

    n_reg = d + 2 * n_obs + 1  # total number of registers

    control = QuantumRegister(d, 'd')
    data = QuantumRegister(n_obs, 'x')
    labels = QuantumRegister(n_obs, 'y')
    data_test = QuantumRegister(1, 'x^{test}')
    label_test = QuantumRegister(1, 'y^{test}')
    c = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(control, data, labels, data_test, label_test, c)

    qc.initialize(x_test, [data_test[0]])

    for index in range(n_obs):
        qc.initialize(X_data[index], [data[index]])
        qc.initialize(Y_data[index], [labels[index]])

    for i in range(d):
        qc.h(control[i])

    U1 = [0, 2]  # np.random.choice(range(4), 2, replace=False)
    U2 = [1, 3]  # np.random.choice(range(4), 2, replace=False)
    # U3 = [0, 0]  # np.random.choice(range(4), 2, replace=False)
    U4 = [2,3]  # np.random.choice(range(4), 2, replace=False)

    qc.barrier()

    # U1
    qc.cswap(control[0], data[int(U1[0])], data[int(U1[1])])
    qc.cswap(control[0], labels[int(U1[0])], labels[int(U1[1])])

    qc.x(control[0])

    # U2
    qc.cswap(control[0], data[int(U2[0])], data[int(U2[1])])
    qc.cswap(control[0], labels[int(U2[0])], labels[int(U2[1])])

    qc.barrier()

    # U3
    # qc.cswap(control[1], data[int(U3[0])], data[int(U3[1])])
    # qc.cswap(control[1], labels[int(U3[0])], labels[int(U3[1])])

    qc.x(control[1])

    # U4
    qc.cswap(control[1], data[int(U4[0])], data[int(U4[1])])
    qc.cswap(control[1], labels[int(U4[0])], labels[int(U4[1])])

    # qc.barrier()
    # qc.initialize(x_test, [data_test[0]])
    # qc.barrier()

    # C
    ix_cls = 3
    # qc.h(labels[ix_cls])
    # qc.cswap(labels[ix_cls], data[ix_cls], test[0])
    # qc.h(labels[ix_cls])
    # qc.measure(labels[ix_cls], c)
    qc.h(label_test[0])
    qc.cswap(label_test[0], data[ix_cls], data_test[0])
    qc.h(label_test[0])
    qc.cx(labels[ix_cls], label_test[0])
    qc.measure(label_test[0], c)
    return qc



def ensemble_random_swap(X_data, Y_data, x_test, d = 2 ):
    # d = 2  # number of control qubits
    n_obs = len(X_data)

    n_reg = d + 4 * n_obs + 1  # total number of registers

    control = QuantumRegister(d, 'control')
    data = QuantumRegister(n_obs, 'x')
    labels = QuantumRegister(n_obs, 'y')
    ancilla_x = QuantumRegister(n_obs, 'ancilla_x')
    ancilla_y = QuantumRegister(n_obs, 'ancilla_y')
    ancilla_test = QuantumRegister(2, 'ancilla_test')
    data_test = QuantumRegister(1, 'x test')
    label_test = QuantumRegister(1, 'y test')
    c = ClassicalRegister(1)

    qc = QuantumCircuit(control, data, labels, data_test, label_test, ancilla_x, ancilla_y, ancilla_test, c)

    for index in range(n_obs):
        qc.initialize(X_data[index], [data[index]])
        qc.initialize(Y_data[index], [labels[index]])


    qc.initialize(x_test, [data_test[0]])


    for i in range(d):
        qc.h(control[i])

    U1 = np.random.choice(range(n_obs), 2, replace=False)
    U2 = np.random.choice(range(n_obs), 2, replace=False)
    U3 = np.random.choice(range(n_obs), 2, replace=False)
    U4 = np.random.choice(range(n_obs), 2, replace=False)

    qc.barrier()

    for i in range(n_obs):
        qc.cswap(control[0], data[i], ancilla_x[i])
        qc.cswap(control[0], labels[i], ancilla_y[i])
    qc.cswap(control[0], data_test[0], ancilla_test[0])
    qc.cswap(control[0], label_test[0], ancilla_test[1])

    qc.barrier()

    # U1
    qc.swap(data[int(U1[0])], data[int(U1[1])])
    qc.swap(labels[int(U1[0])], labels[int(U1[1])])


    # U2
    qc.swap(ancilla_x[int(U2[0])], ancilla_x[int(U2[1])])
    qc.swap(ancilla_y[int(U2[0])], ancilla_x[int(U2[1])])

    qc.barrier()

    for i in range(n_obs):
        qc.cswap(control[0], data[i], ancilla_x[i])
        qc.cswap(control[0], labels[i], ancilla_y[i])
    qc.cswap(control[0], data_test[0], ancilla_test[0])
    qc.cswap(control[0], label_test[0], ancilla_test[1])

    qc.barrier()

    for i in range(n_obs):
        qc.cswap(control[1], data[i], ancilla_x[i])
        qc.cswap(control[1], labels[i], ancilla_y[i])
    qc.cswap(control[1], data_test[0], ancilla_test[0])
    qc.cswap(control[1], label_test[0], ancilla_test[1])

    qc.barrier()

    # U3
    qc.swap(data[int(U3[0])], data[int(U3[1])])
    qc.swap(labels[int(U3[0])], labels[int(U3[1])])

    # U4
    qc.swap(ancilla_x[int(U4[0])], ancilla_x[int(U4[1])])
    qc.swap(ancilla_y[int(U4[0])], ancilla_y[int(U4[1])])

    qc.barrier()

    for i in range(n_obs):
        qc.cswap(control[1], data[i], ancilla_x[i])
        qc.cswap(control[1], labels[i], ancilla_y[i])
    qc.cswap(control[1], data_test[0], ancilla_test[0])
    qc.cswap(control[1], label_test[0], ancilla_test[1])

    qc.barrier()

    # C
    ix_cls = 3

    qc.h(label_test[0])
    qc.cswap(label_test[0], data[ix_cls], data_test[0])
    qc.h(label_test[0])
    qc.cx(labels[ix_cls], label_test[0])
    qc.measure(label_test[0], c)
    return qc


def exec_simulator(qc, n_shots = 8192):
    # QASM simulation
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots = n_shots)
    results = job.result()
    answer = results.get_counts(qc)
    return answer
