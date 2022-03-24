# Import packages and functions
import os.path
import sys
sys.path.insert(1, '../')

from Utils import *
from modeling import *



def plot_multiple_experiments(runs, avg_sim, avg_real, ens_sim, ens_real, filename='comparison_avg_quantum', folder ='output'):
    x = np.arange(runs)

    ax = plt.subplot()
    ax.plot(x, ens_sim, color='orange', label='Quantum Ensemble (simulator)', zorder=1, linewidth=5)
    ax.scatter(x, avg_sim, label='Quantum Ensemble (Simulator)', color='sienna', zorder=2, linewidth=.5)

    ax.plot(x, ens_real, color='deepskyblue', label='Quantum Ensemble ($ibmq\_montreal^*$)', zorder=1, linewidth=5)
    ax.scatter(x, avg_real, label='Classical Ensemble ($ibmq\_casablanca$)', color='steelblue', zorder=2, linewidth=.5)



    ax.grid(alpha=0.3)
    #ax.set_title('Comparison', size=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.2),
              ncol=3, fancybox=True, shadow=True, fontsize=9)
    filepath=os.path.join(folder, filename)
    plt.savefig(filepath + '.png', dpi=300, bbox_inches='tight')
    plt.show()
#plot_multiple_experiments(n, avg, ens_sim, ens_real)



def init_quantum_devices(real = 'ibm_lagos', fake= 'fake_montreal', simulator='ibmq_qasm_simulator'):
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-research')
    backend = provider.get_backend(real)

    backend_sim = provider.get_backend(simulator)

    provider = FakeProvider()
    backend_fake = provider.get_backend(fake)

    return backend, backend_fake, backend_sim


def run_real_device(qc, backend, shots=8192):
    job = execute(qc, backend, shots=shots)
    results = job.result()
    r = results.get_counts(qc)
    return r


def quantum_average_run(n_runs=20, seed=543, n_shots = 8192):

    backend, backend_fake, backend_sim = init_quantum_devices(real='ibmq_casablanca', fake= 'fake_montreal',
                                                              simulator='ibmq_qasm_simulator')

    full_data = []

    avg = []
    avg_sim =[]
    ens_sim = []
    ens_real = []

    y_labels =[[0,1], [1,0]]

    print('Executing Cosine Classifier ...............')

    np.random.seed(seed)
    for run in np.arange(n_runs):
        print('classical run: ', run)
        x1 = [np.random.randint(1, 9), np.random.randint(1, 9)]
        x2 = [np.random.randint(1, 9), np.random.randint(1, 9)]
        x3 = [np.random.randint(1, 9), np.random.randint(1, 9)]
        x4 = [np.random.randint(1, 9), np.random.randint(1, 9)]

        y1 = y_labels[np.random.randint(0, 2)]
        y2 = y_labels[np.random.randint(0, 2)]
        y3 = y_labels[np.random.randint(0, 2)]
        y4 = y_labels[np.random.randint(0, 2)]

        Y_data = [y1, y2, y3, y4]
        X_data = [x1, x2, x3, x4]

        x_test = [np.random.randint(1, 9), np.random.randint(1, 9)]

        X_data, Y_data, x_test = load_data_custom(X_data, Y_data, x_test=x_test)

        qc1 = quantum_cosine_classifier(X_data[0], x_test, Y_data[0])
        r1 = run_real_device(qc1, backend, n_shots)
        r1 = retrieve_proba(r1)

        r1_sim = run_real_device(qc1, backend_sim, n_shots)
        r1_sim = retrieve_proba(r1_sim)


        qc2 = quantum_cosine_classifier(X_data[1], x_test, Y_data[1])
        r2 = run_real_device(qc2, backend, n_shots)
        r2 = retrieve_proba(r2)

        r2_sim = run_real_device(qc2, backend_sim, n_shots)
        r2_sim = retrieve_proba(r2_sim)


        qc3 = quantum_cosine_classifier(X_data[2], x_test, Y_data[2])
        r3 = run_real_device(qc3, backend, n_shots)
        r3 = retrieve_proba(r3)

        r3_sim = run_real_device(qc3, backend_sim, n_shots)
        r3_sim = retrieve_proba(r3_sim)


        qc4 = quantum_cosine_classifier(X_data[3], x_test, Y_data[3])
        r4 = run_real_device(qc4, backend, n_shots)
        r4 = retrieve_proba(r4)

        r4_sim = run_real_device(qc4, backend_sim, n_shots)
        r4_sim = retrieve_proba(r4_sim)


        out = [r1, r2, r3, r4]
        out_sim = [r1_sim, r2_sim, r3_sim, r4_sim]

        p0 = [p[0] for p in out]
        p1 = [p[1] for p in out]

        p0_sim = [p[0] for p in out_sim]
        p1_sim = [p[1] for p in out_sim]


        r_avg = [np.mean(p0), np.mean(p1)]
        avg.append(r_avg[1])

        r_avg_sim = [np.mean(p0_sim), np.mean(p1_sim)]
        avg_sim.append(r_avg_sim[1])

        full_data.append([X_data, Y_data, x_test])



    print('Executing Ensemble ...............')
    for dataset in full_data:
        print('quantum run')
        X_data, Y_data, x_test = dataset[0], dataset[1], dataset[2]

        qc = ensemble_fixed_U(X_data, Y_data, x_test)
        qc = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)

        r_real = run_real_device(qc, backend_fake, n_shots)
        r_ens_real = retrieve_proba(r_real)

        r_sim = run_real_device(qc, backend_sim, n_shots)
        r_ens_sim = retrieve_proba(r_sim)

        ens_sim.append(r_ens_sim[1])
        ens_real.append(r_ens_real[1])

    return avg_sim, avg, ens_sim, ens_real


n = 20
folder = 'output'
create_dir(folder)

avg_sim, avg, ens_sim, ens_real = quantum_average_run(n_runs=n)

df = pd.DataFrame([avg_sim, avg, ens_sim, ens_real]).transpose()
df.columns = ['classical_sim', 'classical_real', 'ensemble_simulator', 'ensemble_real_device']
df.to_csv(folder+'/ens_avg_cosine_result.csv', index=False)

plot_multiple_experiments(n, avg_sim, avg, ens_sim, ens_real)



