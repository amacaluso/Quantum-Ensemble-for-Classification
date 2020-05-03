from Utils import *
import sys
sys.path.insert(1, '../')


import random
# seed = random.randint(1, 10000)
seed = 4552
std = .20
test_size = .2

np.random.seed(seed)


# Simulated data
from sklearn import datasets
X, y = datasets.make_blobs(n_samples=200, centers=[[0.5, .1],[.1, 0.5]],
                           n_features=2, center_box=(0, 1),
                           cluster_std = std, random_state=seed)
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'g^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.grid()
plt.show()

columns = ['$x_1$', '$x_2$', 'Y']
data = pd.concat( [pd.DataFrame(X), pd.DataFrame(np.where(y == 0, 'class 0', 'class 1'))], axis=1)
data.columns = columns
data.to_csv('data/all_data.csv', index = False)


multivariateGrid('$x_1$', '$x_2$', 'Y', df=data)



Y = []
for el in y:
    if el == 0:
        Y.append([1,0])
    else:
        Y.append([0,1])

Y = np.asarray(Y)

accuracy = []

from sklearn.model_selection import train_test_split
X_train, X_test, Y_vector_train, Y_vector_test, y_train, y_test = train_test_split(X, Y, y,
                                                                                   random_state=seed,
                                                                                   test_size=test_size)


print("Size Training Set: ", len(X_train))
print("Size Test Set: ", len(X_test))


output = 'output'
folder_img = 'IMG'

create_dir(output)
create_dir(folder_img)

n_shots = 1000


for i in range(10):

    ''' State Preparation'''
    ## ++++++++++++++++++++++++++++++++++ ##
    # Training Set

    n = range(len(X_train))
    # ix = np.random.choice(n, 1)[0]

    TP = 0
    predictions = []
    probabilities = []
    for x_test, y_ts in zip(X_test, Y_vector_test):
        ix = np.random.choice(n, 1)[0]

        x_train = X_train[ix]; x_tr = normalize_custom(x_train)
        y_tr = Y_vector_train[ix]

        x_ts = normalize_custom(x_test)
        qc = cos_classifier(x_tr, x_ts, y_tr)
        r = exec_simulator(qc, n_shots=n_shots)

        if '0' not in r.keys():
            r['0'] = 0
        elif '1' not in r.keys():
            r['1'] = 0

        p0 = (r['0'] / (r['0'] + r['1']))
        p0 = p0
        p1 = 1 - p0

        predictions.append(r)
        probabilities.append(predict_cos(r))
        probs = [p0, p1]
        print('output:', r, probs)

    #    print(predict_cos(r))
        if predict_cos(r)[0] > predict_cos(r)[1]:
            pred = [1,0]
            pred = np.asarray(pred)
        else:
            pred = [0, 1]
            pred = np.asarray(pred)

        #print(predict_cos(r), y_tr)
        print('Data:', 'train=', x_train, 'test=', x_test)
        print('results: ', 'tr:', y_tr, 'ts:', y_ts, 'pred:', pred, '\n')

        if np.array_equal(pred, y_ts):
            TP = TP+1

    print('True positive=', TP/len(X_test))
    accuracy.append(TP/len(X_test))

print(np.mean(accuracy))
print(np.std(accuracy))


from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
provider.backends()
real_device = provider.get_backend('ibmq_qasm_simulator')

#
#
predictions = []
for x_ts, y_ts in zip(X_test, Y_vector_test):
    ix_y1 = np.random.choice(np.where(y_train == 1)[0], 2, replace=False)
    ix_y0 = np.random.choice(np.where(y_train == 0)[0], 2, replace=False)

    data = np.concatenate([X_train[ix_y1], X_train[ix_y0]])

    for i in range(len(data)):
        data[i] = normalize_custom(data[i])

    x_ts = normalize_custom(x_ts)

    labels = np.concatenate([Y_vector_train[ix_y1], Y_vector_train[ix_y0]])

    qc = ensemble(data, labels, x_ts)
    r = exec_simulator(qc, n_shots=8192)

    predictions.append(retrieve_proba(r))

    print(retrieve_proba(r), y_ts)


import sklearn
from sklearn.metrics import brier_score_loss, accuracy_score

predicted_class =  np.round(np.asarray(predictions))
acc = accuracy_score(np.array(predicted_class)[:, 1],
                                     np.array(Y_vector_test)[:, 1])

columns = ['X1', 'X2', 'class0', 'class1']
train_data = pd.concat( [pd.DataFrame(X_train), pd.DataFrame(Y_vector_train)], axis=1)
train_data.columns = columns
train_data.to_csv('data/train_data.csv', index = False)

test_data = pd.concat([pd.DataFrame(X_test), pd.DataFrame(Y_vector_test)], axis=1)
p0 = [p[0] for p in predictions]
p1 = [p[1] for p in predictions]

test_data['p0'] = p0
test_data['p1'] = p1
test_data['predicted_class'] = [pred[1] for pred in predicted_class]

test_data.columns=columns + ['p0','p1','predicted_class']

test_data.to_csv('results/test_data.csv', index=False)


brier = brier_score_loss(y, p1)

print('Accuracy=',acc)
print('Brier score=', brier)

#
# T = 0
# for y_pred, y_true in zip(predictions, Y_vector_test):
#     if sum(np.round(np.asarray(y_pred))-y_true)==0:
#         T = T+1
# T/len(Y_vector_test)
# (np.round(np.asarray(predictions)) - Y_vector_test)

# qc_ens_v2 = qc_ensemble_v2(D, x_test)
# #%%
# ## Classifier 1
# r1 = exec_simulator(qc1)
# save_dict( r1, name = 'c1' )
# #%%
# ## Classifier 2
# r2 = exec_simulator(qc2)
# save_dict( r2, name = 'c2' )
# #%%
# ## Classifier 3
# r3 = exec_simulator(qc3)
# save_dict( r3, name = 'c3' )
# #%%
# ## Classifier 4
# r4 = exec_simulator(qc4)
# save_dict( r4, name = 'c4' )
# #%%
# ## Ensemble original
# r_ens = exec_simulator(qc_ens)
# save_dict( r_ens, name = 'ensemble' )
# #%%
# ## Ensemble v2
# r_ens_v2 = exec_simulator(qc_ens_v2)
# save_dict( r_ens_v2, name = 'ensemble_v2' )
# #%%
# #diagram =
# qc_ens_v2.draw(output="mpl")
# #%%
# print(r1, r2, r3, r4, r_ens, r_ens_v2)
# #%% md
# # Circuits to run
#
#
#
#
#
#
#
#
#
#
#























#
#
# ''' State Preparation'''
# ## ++++++++++++++++++++++++++++++++++ ##
#
# # Training points
# x_train = [3, -1 ]; x_train = normalize_custom(x_train)
# #y_train = [1, 0]
#
# # Test point
# x_test = [2, 2]; x_test = normalize_custom( x_test )
#
# # Visualization of the two vectors
# V = np.array([ x_train, x_test])
# origin = [0], [0] # origin point
# plt.quiver(*origin, V[:,0], V[:,1], color=['r','b'], scale = 4)
# plt.show()
#
#
# # Quantum Circuit for Cosine-distance classifier
# c = ClassicalRegister(1)
# ancilla = QuantumRegister( 1 , 'y\_test}')
#
# x_tr = QuantumRegister(1, 'x\_train')
# x_ts = QuantumRegister(1, 'x\_test')
# y_tr = QuantumRegister(1, 'y\_train')
#
# qc = QuantumCircuit( x_tr, x_ts, ancilla, y_tr, c)
#
# x_tr = x_tr[0]
# y_tr = y_tr[0]
# x_ts = x_ts[0]
#
# qc.initialize(x_train, [ x_tr ])
# qc.initialize(x_test, [ x_ts ])
# qc.x( y_tr )
# qc.barrier()
#
# qc.h( ancilla )
# qc.cswap( ancilla, x_ts, x_tr )
# qc.h(ancilla)
# qc.barrier()
#
# qc.cx( y_tr, ancilla)
# qc.barrier()
#
# qc.measure(ancilla, c[0])
# print(qc)
#
# # diagram = qc.draw(output="mpl")
# # diagram.show()
# # diagram.savefig(folder_img + "/cosine_classifier.jpeg",
# #                 format="jpeg")
#
#
# # QASM Simulation
# sim_backend = BasicAer.get_backend('qasm_simulator')
# job = execute(qc, sim_backend, shots=8192)
# results = job.result()
# answer = results.get_counts(qc)
# print(answer)
#
# if len(answer) == 1:
#     quantum_prob = 1
# else:
#     quantum_prob = answer['0']/(answer['0'] + answer['1'] )
#
# answer_sim_p0 = quantum_prob
# answer_sim_p1 = 1 - quantum_prob
#
#
# ## Running on real device
#
# # ****** ---------------------------------------------******* #
# job = execute(qc, backend, shots=8192)
# results = job.result()
# answer = results.get_counts(qc)
# print(answer)
#
# if len(answer) == 1:
#     quantum_prob = 1
# else:
#     quantum_prob = answer['0']/(answer['0'] + answer['1'] )
#
# answer_rl_p0 = quantum_prob
# answer_rl_p1 = 1- quantum_prob
#
#
# N = 2
# p0 = [ answer_sim_p0, answer_rl_p0]
# p1 = [ answer_sim_p1, answer_rl_p1]
# #
# fig, ax = plt.subplots()
# #
# ind = np.arange(N)  # the x locations for the groups
# width = 0.35  # the width of the bars
# pl1 = ax.bar(ind, p0, width, bottom=0)
# pl2 = ax.bar(ind + width, p1, width, bottom=0)
# #
# ax.set_title('Test point classifications')
# ax.set_xticks(ind + width / 2)
# ax.set_xticklabels(('QASM', 'REAL DEVICE'), rotation = 45)
# #
# ax.legend((pl1[0], pl2[0]), ('P(y=0)', 'P(y=1)'))
# ax.autoscale_view()
# #
# plt.savefig(folder_img + '/qasm_vs_real.png',
#                 format="jpeg",  bbox_inches="tight")
# plt.show()
#
#
#
# # def execution_qc( qc, device =0, n_shots = 1000):
# #     # device = 0 --> SIMULATOR
# #     # device = 1 --> REAL DEVICE
# #     if device == 0:
# #         backend = BasicAer.get_backend('qasm_simulator')
# #         job = execute(qc, backend, shots=n_shots)
# #         results = job.result()
# #      answer = results.get_counts(qc)
# #         # print(answer)
# #         # if len(answer) == 1:
# #         #     quantum_prob = 1
# #         # else:
# #         #     quantum_prob = answer['0']/(answer['0'] + answer['1'] )
# #
# #         answer_sim_p0 = quantum_prob
# #         answer_sim_p1 = 1 - quantum_prob
# #         '''QASM simulator'''
# #         print( 'P(y_test = 0) =' ,quantum_prob )
# #         print( 'P(y_test = 1) =' ,1 - quantum_prob )
