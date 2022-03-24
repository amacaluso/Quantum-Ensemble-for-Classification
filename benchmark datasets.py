import os.path

from Utils import *
from modeling import *
from import_data import *
from qiskit.test.mock import FakeProvider

def run_cosine_classifier(X_train, X_test, Y_train, Y_test, seed, backend, n_shots = 8192):
    predictions = []
    n = len(X_train)
    test_size = len(X_test)/(len(X_test)+len(X_train))

    Y_vector_train = label_to_array(Y_train)
    Y_vector_test = label_to_array(Y_test)

    np.random.seed(seed)
    for x_test, y_ts in zip(X_test, Y_vector_test):
        ix = np.random.choice(int(n * (1 - test_size)), 1)[0]
        x_train = X_train[ix]
        x_train = normalize_custom(x_train)
        y_train = Y_vector_train[ix]

        x_test = normalize_custom(x_test)

        qc = cos_classifier(x_train, x_test, y_train)
        job = execute(qc, backend, shots=n_shots)
        results = job.result()
        r = results.get_counts(qc)

        predictions.append(retrieve_proba(r))

    accuracy, brier = evaluation_metrics(predictions, X_test, Y_test, save=False)
    print('seed: {} | n_train: {} | d: {} | Acc: {} | Brier: {} | balanced: {}'.format(seed, 1, 0, accuracy, brier, False))
    return accuracy, brier




def run_ensemble(X_train, X_test, Y_train, Y_test, d, n_train, seed, backend, n_shots = 8192, balanced = True):

    predictions = []
    for x_test, y_ts in zip(X_test, Y_test):
        X_data, Y_data = training_set(X_train, Y_train, n=n_train, seed=seed)
        x_test = normalize_custom(x_test)

        qc = ensemble(X_data, Y_data, x_test, n_swap=1, d=d, balanced=balanced)
        qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=1)

        job = execute(qc, backend, shots=n_shots)
        results = job.result()
        r = results.get_counts(qc)

        predictions.append(retrieve_proba(r))

    accuracy, brier = evaluation_metrics(predictions, X_test, Y_test, save=False)
    print('seed:', seed, '| n_train:', n_train,'| d:', d, '| Acc:', accuracy, '| Brier:', brier, '| balanced:', balanced)
    return accuracy, brier


# ### Bivariate distribution

# In[2]:


def run_gaussian(backend, test_size=.2, seeds=[123], d_vector = range(1,5), quantum_label = 'ibmq_qasm_simulator', folder='output'):
    X,Y = load_bivariate_gaussian(n_train=100, plot=False)
    dataset = 'gaussian'

    results = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, test_size=test_size)

        accuracy, brier = run_cosine_classifier(X_train, X_test, y_train, y_test, seed, backend, n_shots=8192)
        row_bal = [dataset, 0, 1, test_size, seed, accuracy, brier, False, quantum_label]
        results.append(row_bal)

        for d in d_vector:
            n_train = 2**d
            if n_train > 8:
                n_train = 8
            
            np.random.seed(seed)
            accuracy, brier = run_ensemble(X_train, X_test, y_train, y_test, d, n_train, seed, backend = backend)
            row_bal = [dataset, d, n_train, test_size, seed, accuracy, brier, True, quantum_label]

            results.append(row_bal)

        print(seed, '-------------------------------------------')

    data = pd.DataFrame(results)
    data.columns = ['dataset', 'd', 'n_train', 'test_size', 'seed', 'accuracy', 'brier', 'balanced', 'quantum_label']
    filepath = os.path.join(folder, quantum_label, dataset)
    data.to_csv(filepath + '.csv', index=False)

    print('Gaussian done \n')
    return results





def run_iris_1_vs_2(backend, test_size=.2, seeds=[123], d_vector = range(1,5), quantum_label = 'ibmq_qasm_simulator', folder='output'):
    X,Y = load_iris(0)
    dataset = 'iris_class1_vs_class2'

    results = []


    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, test_size=test_size)

        accuracy, brier = run_cosine_classifier(X_train, X_test, y_train, y_test, seed, backend, n_shots=8192)
        row_bal = [dataset, 0, 1, test_size, seed, accuracy, brier, False, quantum_label]
        results.append(row_bal)

        for d in d_vector:
            n_train = 2**d
            if n_train > 8:
                n_train = 8
            np.random.seed(seed)
            accuracy, brier = run_ensemble(X_train, X_test, y_train, y_test, d, n_train, seed, backend = backend)
            row_bal = [dataset, d, n_train, test_size, seed, accuracy, brier, True, quantum_label]
            results.append(row_bal)

        print(seed, '-------------------------------------------')

    data = pd.DataFrame(results)
    data.columns = ['dataset', 'd', 'n_train', 'test_size', 'seed', 'accuracy', 'brier', 'balanced', 'quantum_label']
    filepath = os.path.join(folder, quantum_label, dataset)
    data.to_csv(filepath + '.csv', index=False)

    print('Iris done \n')
    return results




def run_iris_0_vs_2(backend, test_size=.2, seeds=[123], d_vector = range(1,5), quantum_label = 'ibmq_qasm_simulator', folder='output'):

    X,Y = load_iris(1)
    dataset = 'iris_class0_vs_class2'

    results = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, test_size=test_size)

        accuracy, brier = run_cosine_classifier(X_train, X_test, y_train, y_test, seed, backend, n_shots=8192)
        row_bal = [dataset, 0, 1, test_size, seed, accuracy, brier, False, quantum_label]
        results.append(row_bal)

        for d in d_vector:
            n_train = 2**d
            if n_train > 8:
                n_train = 8
            np.random.seed(seed)
            accuracy, brier = run_ensemble(X_train, X_test, y_train, y_test, d, n_train, seed, backend = backend)
            row_bal = [dataset, d, n_train, test_size, seed, accuracy, brier, True, quantum_label]
            results.append(row_bal)

        print(seed, '-------------------------------------------')

    data = pd.DataFrame(results)
    data.columns = ['dataset', 'd', 'n_train', 'test_size', 'seed', 'accuracy', 'brier', 'balanced', 'quantum_label']
    filepath = os.path.join(folder, quantum_label, dataset)
    data.to_csv(filepath + '.csv', index=False)

    print('Iris done \n')
    return results


# In[5]:


def run_iris_0_vs_1(backend, test_size=.2, seeds=[123], d_vector = range(1,5), quantum_label = 'ibmq_qasm_simulator', folder='output'):
    X,Y = load_iris(2)

    results = []
    dataset = 'iris_class0_vs_class1'

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, test_size=test_size)

        accuracy, brier = run_cosine_classifier(X_train, X_test, y_train, y_test, seed, backend, n_shots=8192)
        row_bal = [dataset, 0, 1, test_size, seed, accuracy, brier, False, quantum_label]
        results.append(row_bal)

        for d in d_vector:

            n_train = 2**d
            if n_train > 8:
                n_train = 8
            np.random.seed(seed)
            accuracy, brier = run_ensemble(X_train, X_test, y_train, y_test, d, n_train, seed, backend = backend)
            row_bal = [dataset, d, n_train, test_size, seed, accuracy, brier, True, quantum_label]
            results.append(row_bal)

        print(seed, '-------------------------------------------')

    data = pd.DataFrame(results)
    data.columns = ['dataset', 'd', 'n_train', 'test_size', 'seed', 'accuracy', 'brier', 'balanced', 'quantum_label']
    filepath = os.path.join(folder, quantum_label, dataset)
    data.to_csv(filepath + '.csv', index=False)

    print('Iris done \n')
    return results



def run_MNIST(backend, test_size=.2, seeds=[123], d_vector = range(1,5), quantum_label = 'ibmq_qasm_simulator', folder='output'):
    results = []
    dataset = 'MNIST'

    for seed in seeds:
        X_train, X_test, y_train, y_test = load_MNIST(n=200, train_size=1-test_size, seed=seed)

        accuracy, brier = run_cosine_classifier(X_train, X_test, y_train, y_test, seed, backend, n_shots=8192)
        row_bal = [dataset, 0, 1, test_size, seed, accuracy, brier, False, quantum_label]
        results.append(row_bal)

        for d in d_vector:
            n_train = 2**d
            if n_train > 8:
                n_train = 8
            np.random.seed(seed)
            accuracy, brier = run_ensemble(X_train, X_test, y_train, y_test, d, n_train, seed, backend = backend)
            row_bal = [dataset, d, n_train, test_size, seed, accuracy, brier, True, quantum_label]
            results.append(row_bal)

        print(seed, '-------------------------------------------')

    data = pd.DataFrame(results)
    data.columns = ['dataset', 'd', 'n_train', 'test_size', 'seed', 'accuracy', 'brier', 'balanced', 'quantum_label']
    filepath = os.path.join(folder, quantum_label, dataset)
    data.to_csv(filepath + '.csv', index=False)

    print('MNIST done \n')
    return results




def run_all(simulator=True, real=False, fake = False, folder='output'): #'ibm_lagos' 'ibm_guadalupe'

    # load IBMQ account and backend
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-research')

    if real == True:
        computer = 'ibm_lagos'
        backend = provider.get_backend(computer)

    if simulator==True:
        computer = 'ibmq_qasm_simulator'
        backend = provider.get_backend(computer)

    if fake == True:
        provider = FakeProvider()
        computer = 'fake_montreal'
        backend = provider.get_backend(computer)

    dir = folder
    if not os.path.exists(dir):
        os.makedirs(dir)

    create_dir(os.path.join(folder,computer))
    # Parameter
    results = []
    test_size=.1
    seeds=list(range(0,10))

    d_vector = [1, 2, 3]#, 4] #list(range(1,4))

    results.append(run_MNIST(backend, test_size, seeds, d_vector, computer))
    #results.append(run_gaussian(backend, test_size, seeds, d_vector, computer))
    results.append(run_iris_0_vs_1(backend, test_size, seeds, d_vector, computer))
    results.append(run_iris_0_vs_2(backend, test_size, seeds, d_vector, computer))
    results.append(run_iris_1_vs_2(backend, test_size, seeds, d_vector, computer))
    #results.append(run_breast(backend, test_size, seeds, d_vector, computer))

    data = []
    for df in results:
        for row in df:
            data.append(row)

    data = pd.DataFrame(data)
    data.columns = ['dataset', 'd', 'n_train', 'test_size', 'seed', 'accuracy', 'brier', 'balanced', 'device']

    filepath = os.path.join(folder, 'ensemble_results.csv')

    if not os.path.isfile(filepath):
        data.to_csv(filepath, header='column_names')
    else:  # else it exists so append without writing the header
        data.to_csv(filepath, mode='a', header=False)

    return data

df = run_all(simulator=True, real=False, fake= False)
#df = run_all(simulator=False, real=False, fake=True)