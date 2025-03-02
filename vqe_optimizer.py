from qubots.base_optimizer import BaseOptimizer
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class VQEOptimizer(BaseOptimizer):
    def __init__(self, num_layers=2, max_iters=100, learning_rate=0.1, n_shots=1000, verbose=False):
        self.num_layers = num_layers
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.n_shots = n_shots
        self.verbose = verbose

    def optimize(self, problem, initial_solution=None, **kwargs):
        # Retrieve Ising parameters directly from problem
        h_dict, J_dict, offset, edges = problem.h, problem.J, problem.offset, problem.edges

        n = len(h_dict)

        coeffs = []
        obs = []
        for i in range(n):
            if (i,) in h_dict and abs(h_dict[(i,)]) > 1e-8:
                coeffs.append(h_dict[(i,)])
                obs.append(qml.PauliZ(i))
        for (i, j) in edges:
            if (i, j) in J_dict and abs(J_dict[(i, j)]) > 1e-8:
                coeffs.append(J_dict[(i, j)])
                obs.append(qml.operation.Tensor(qml.PauliZ(i), qml.PauliZ(j)))

        H_C = qml.Hamiltonian(coeffs, obs)

        p = self.num_layers
        dev = qml.device("lightning.qubit", wires=n, shots=self.n_shots)

        def vqe_ansatz(params):
            params = pnp.reshape(params, (p, n))
            for i in range(n):
                qml.Hadamard(wires=i)
            for layer in range(p):
                for i in range(n):
                    qml.RY(params[layer, i], wires=i)
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(H_C)

        vqe_qnode = qml.QNode(vqe_ansatz, dev)
        params = pnp.array(np.random.uniform(0, 2*np.pi, p * n), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)

        for it in range(self.max_iters):
            params, cost = opt.step_and_cost(vqe_qnode, params)
            if self.verbose:
                print(f"VQE Iteration {it}: cost = {cost}")

        @qml.qnode(dev)
        def sample_vqe(params):
            params = pnp.reshape(params, (p, n))
            for i in range(n):
                qml.Hadamard(wires=i)
            for layer in range(p):
                for i in range(n):
                    qml.RY(params[layer, i], wires=i)
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.probs(wires=range(n))

        probs = sample_vqe(params)
        state_index = int(np.argmax(probs))
        bitstring = np.array(list(np.binary_repr(state_index, width=n))).astype(np.int8)

        final_cost = problem.evaluate_solution(bitstring.tolist())
        return bitstring.tolist(), final_cost
