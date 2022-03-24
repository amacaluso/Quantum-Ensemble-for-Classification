#!/usr/bin/env python
# coding: utf-8

#Import modules and packages
import sys
sys.path.insert(1, '../')

from Utils import *
from modeling import * 



def quantum_cosine_classifier(train, test, label_train):
    # x_train = train
    # x_new = test
    # y_train = label_train
    c = ClassicalRegister(1, 'c')
    x_train = QuantumRegister(1, 'x_{b}')
    x_test = QuantumRegister(1, 'x^{(test)}')
    y_train = QuantumRegister(1, 'y_{b}')
    y_test = QuantumRegister(1, 'y^{(test)}')
    qc = QuantumCircuit(x_train, x_test, y_train, y_test, c)
    
    S1 = state_prep(train)
    qc.unitary(S1, [0], label='$S_{x}$')

    S2 = state_prep(test)
    qc.unitary(S2, [1], label='$S_{x}$')
    
    S2 = state_prep(label_train)
    qc.unitary(S2, [2], label='$S_{y}$')

    #qc.initialize(label_train, [y_train[0]])
    qc.barrier()
    qc.h(y_test)
    qc.cswap(y_test, x_train, x_test)
    qc.h(y_test)
    qc.barrier()
    qc.cx(y_train, y_test)
    qc.measure(y_test, c)
    return qc    
 

def exec_circuit(qc, backend_name = 'ibmq_qasm_simulator', n_shots = 8192):
    IBMQ.load_account()

    provider = IBMQ.get_provider(hub='ibm-q-research')
    provider.backends()
    backend = provider.get_backend(backend_name)

    job = execute(qc, backend, shots = n_shots)
    results = job.result()
    answer = results.get_counts(qc)
    return answer


n=500
n_shots=8192

computer = 'ibm_lagos'

x = []
x_err = []
P0 = []
P1 = []

for i in np.arange(n):
    print('cosine classifier: ', i)
    '''Random generated dataset'''
    x_train = [random.uniform(-1, 1), random.uniform(0, 1)]
    x_train_norm = normalize_custom(x_train)
    x_test = [random.uniform(-1, 1), random.uniform(0, 1)]
    x_test_norm = normalize_custom(x_test)

    '''Compute cosine distance and append it to x'''
    d_cos = cosine_similarity([x_train], [x_test])[0][0]
    x.append(d_cos)

    '''If x_train belongs to class 0'''
    qc = quantum_cosine_classifier(x_train_norm, x_test_norm, [1,0])


    #r = exec_circuit(qc, backend_name = 'ibm_lagos', n_shots=n_shots)
    r = exec_circuit(qc, backend_name=computer, n_shots=n_shots)

    P_q = r['0']/n_shots
    P0.append(P_q)
    d_cos_err = np.sqrt(2*P_q-1)
    x_err.append( d_cos_err )


P0 = np.array(P0)
x = np.array(x)
x_err = np.array(x_err)

order = x.argsort()

x = x[order[::-1]]
P0 = P0[order[::-1]]
x_err = x_err[order[::-1]]

err = [abs(x1 - x2) for (x1, x2) in zip(abs(x), x_err)]
err = np.array(err)
P1 = 1-P0

df = pd.DataFrame([x, P0, P1, err]).transpose()
df.to_csv('output/' + computer + '_cosine_result_behaviour.csv', index=False)

quantum_cos_random_data(x, P0, P1, err, save = True, folder='output', computer=computer)
qc.draw(output='mpl', scale=.7)
plt.show()


#exec(open('Quantum Ensemble as Simple Averaging.py').read())


