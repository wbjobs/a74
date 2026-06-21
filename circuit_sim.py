#!/usr/bin/env python3
"""
Circuit Simulator - AC Small Signal + Transient Analysis
Supports R, L, C, independent V sources, 4 types of dependent sources
AC: Modified Nodal Analysis (MNA) with Newton-Raphson in s-domain
Transient: Implicit Trapezoidal Integration in time domain
"""

import sys
import argparse
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


CONTROLLED_TYPES = {'VCC', 'VCV', 'CCC', 'CCV'}
TRANSIENT_SOURCES = {'STEP', 'PULSE', 'SIN'}


def step_signal(t: float, t0: float = 0.0, V0: float = 0.0, V1: float = 1.0) -> float:
    """Unit step: V0 for t < t0, V1 for t >= t0"""
    return V0 if t < t0 else V1


def pulse_signal(t: float, t0: float = 0.0, t1: float = 1e-3,
                 V0: float = 0.0, V1: float = 1.0, period: float = None) -> float:
    """Pulse: V0 -> V1 at t0, V1 -> V0 at t1, optional periodic"""
    if period is not None:
        t = t % period
    if t < t0 or t >= t1:
        return V0
    return V1


def sin_signal(t: float, V0: float = 0.0, Vamp: float = 1.0,
               freq: float = 1000.0, phase: float = 0.0) -> float:
    """Sinusoidal: V(t) = V0 + Vamp * sin(2*pi*freq*t + phase_rad)"""
    phase_rad = np.deg2rad(phase)
    return V0 + Vamp * np.sin(2 * np.pi * freq * t + phase_rad)


class TransientSource:
    def __init__(self, name: str, source_type: str, params: Dict):
        self.name = name
        self.source_type = source_type
        self.params = params
        self._func = self._create_function()

    def _create_function(self) -> Callable[[float], float]:
        if self.source_type == 'STEP':
            t0 = self.params.get('t0', 0.0)
            V0 = self.params.get('v0', 0.0)
            V1 = self.params.get('v1', 1.0)
            return lambda t: step_signal(t, t0, V0, V1)
        elif self.source_type == 'PULSE':
            t0 = self.params.get('t0', 0.0)
            t1 = self.params.get('t1', 1e-3)
            V0 = self.params.get('v0', 0.0)
            V1 = self.params.get('v1', 1.0)
            period = self.params.get('period', None)
            return lambda t: pulse_signal(t, t0, t1, V0, V1, period)
        elif self.source_type == 'SIN':
            V0 = self.params.get('v0', 0.0)
            Vamp = self.params.get('vamp', 1.0)
            freq = self.params.get('freq', 1000.0)
            phase = self.params.get('phase', 0.0)
            return lambda t: sin_signal(t, V0, Vamp, freq, phase)
        else:
            return lambda t: 0.0

    def __call__(self, t: float) -> float:
        return self._func(t)


class Element:
    def __init__(self, name: str, type_: str, n1: int, n2: int, value: float,
                 ac_mag: float = 0.0, ac_phase: float = 0.0,
                 ctrl_n1: Optional[int] = None, ctrl_n2: Optional[int] = None,
                 ctrl_vsource: Optional[str] = None,
                 gain: Optional[float] = None,
                 transient_source: Optional[TransientSource] = None):
        self.name = name
        self.type = type_
        self.n1 = n1
        self.n2 = n2
        self.value = value
        self.ac_mag = ac_mag
        self.ac_phase = ac_phase
        self.ctrl_n1 = ctrl_n1
        self.ctrl_n2 = ctrl_n2
        self.ctrl_vsource = ctrl_vsource
        self.gain = gain
        self.transient_source = transient_source

    def __repr__(self):
        if self.type in CONTROLLED_TYPES:
            return f"{self.name}({self.type}): n{self.n1}-n{self.n2}, gain={self.gain}"
        if self.transient_source:
            return f"{self.name}({self.type}): n{self.n1}-n{self.n2}, {self.transient_source.source_type}"
        return f"{self.name}({self.type}): n{self.n1}-n{self.n2}, val={self.value}"

    def get_voltage(self, t: float) -> float:
        if self.transient_source:
            return self.transient_source(t)
        return self.value


class CircuitParser:
    def __init__(self, filename: str):
        self.filename = filename
        self.elements: List[Element] = []
        self.nodes: set = set()
        self.ac_type: str = "none"
        self.ac_params: Dict = {}
        self.tran_params: Dict = {}
        self.analysis_mode: str = "none"
        self.title: str = ""
        self.newton_params: Dict = {'max_iter': 50, 'tol': 1e-6}

    def parse(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        if lines:
            self.title = lines[0].strip()

        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('*'):
                continue

            parts = line.split()
            if not parts:
                continue

            first = parts[0].lower()

            if first.startswith('.ac'):
                self._parse_ac(parts)
                self.analysis_mode = 'ac'
                continue

            if first.startswith('.tran'):
                self._parse_tran(parts)
                self.analysis_mode = 'tran'
                continue

            if first.startswith('.options'):
                self._parse_options(parts)
                continue

            element = self._parse_element(parts)
            if element:
                self.elements.append(element)
                self.nodes.add(element.n1)
                self.nodes.add(element.n2)
                if element.ctrl_n1 is not None:
                    self.nodes.add(element.ctrl_n1)
                if element.ctrl_n2 is not None:
                    self.nodes.add(element.ctrl_n2)

        if 0 not in self.nodes:
            self.nodes.add(0)

    def _parse_ac(self, parts: List[str]):
        if len(parts) < 2:
            return
        ac_type = parts[1].lower()
        if ac_type == 'sig' and len(parts) >= 4:
            self.ac_type = 'sig'
            self.ac_params = {'sigma': float(parts[2]), 'omega': float(parts[3])}
        elif ac_type == 'lin' and len(parts) >= 5:
            self.ac_type = 'lin'
            self.ac_params = {'npoints': int(parts[2]), 'fstart': float(parts[3]), 'fstop': float(parts[4])}
        elif ac_type == 'single' and len(parts) >= 3:
            self.ac_type = 'single'
            self.ac_params = {'freq': float(parts[2])}

    def _parse_tran(self, parts: List[str]):
        if len(parts) < 3:
            return
        self.tran_params = {
            'tstep': float(parts[1]),
            'tstop': float(parts[2])
        }
        if len(parts) >= 4:
            self.tran_params['tstart'] = float(parts[3])

    def _parse_options(self, parts: List[str]):
        i = 1
        while i < len(parts):
            key = parts[i].lower()
            if key == 'newton_maxiter' and i + 1 < len(parts):
                self.newton_params['max_iter'] = int(parts[i + 1])
                i += 2
            elif key == 'newton_tol' and i + 1 < len(parts):
                self.newton_params['tol'] = float(parts[i + 1])
                i += 2
            else:
                i += 1

    def _parse_element(self, parts: List[str]) -> Optional[Element]:
        name = parts[0]
        type_char = name[0].upper()

        if type_char == 'R':
            if len(parts) < 4:
                return None
            return Element(name, 'R', int(parts[1]), int(parts[2]), float(parts[3]))

        elif type_char == 'C':
            if len(parts) < 4:
                return None
            return Element(name, 'C', int(parts[1]), int(parts[2]), float(parts[3]))

        elif type_char == 'L':
            if len(parts) < 4:
                return None
            return Element(name, 'L', int(parts[1]), int(parts[2]), float(parts[3]))

        elif type_char == 'V':
            if len(parts) < 4:
                return None
            n1, n2 = int(parts[1]), int(parts[2])
            value, ac_mag, ac_phase = 0.0, 0.0, 0.0
            tran_source = None
            i = 3
            while i < len(parts):
                key = parts[i].lower()
                if key == 'dc' and i + 1 < len(parts):
                    value = float(parts[i + 1])
                    i += 2
                elif key == 'ac' and i + 1 < len(parts):
                    ac_mag = float(parts[i + 1])
                    if i + 2 < len(parts):
                        ac_phase = float(parts[i + 2])
                    i += 3
                elif key == 'step' and i + 1 < len(parts):
                    params = {}
                    i += 1
                    while i < len(parts) and '=' in parts[i]:
                        k, v = parts[i].split('=')
                        params[k.lower()] = float(v)
                        i += 1
                    tran_source = TransientSource(name, 'STEP', params)
                elif key == 'pulse' and i + 1 < len(parts):
                    params = {}
                    i += 1
                    while i < len(parts) and '=' in parts[i]:
                        k, v = parts[i].split('=')
                        params[k.lower()] = float(v)
                        i += 1
                    tran_source = TransientSource(name, 'PULSE', params)
                elif key == 'sin' and i + 1 < len(parts):
                    params = {}
                    i += 1
                    while i < len(parts) and '=' in parts[i]:
                        k, v = parts[i].split('=')
                        params[k.lower()] = float(v)
                        i += 1
                    tran_source = TransientSource(name, 'SIN', params)
                else:
                    if value == 0.0:
                        value = float(parts[i])
                    i += 1
            return Element(name, 'V', n1, n2, value, ac_mag, ac_phase,
                          transient_source=tran_source)

        elif type_char == 'G':
            if len(parts) < 6:
                return None
            return Element(name, 'VCC', n1=int(parts[1]), n2=int(parts[2]),
                          value=0.0, ctrl_n1=int(parts[3]), ctrl_n2=int(parts[4]),
                          gain=float(parts[5]))

        elif type_char == 'E':
            if len(parts) < 6:
                return None
            return Element(name, 'VCV', n1=int(parts[1]), n2=int(parts[2]),
                          value=0.0, ctrl_n1=int(parts[3]), ctrl_n2=int(parts[4]),
                          gain=float(parts[5]))

        elif type_char == 'F':
            if len(parts) < 5:
                return None
            return Element(name, 'CCC', n1=int(parts[1]), n2=int(parts[2]),
                          value=0.0, ctrl_vsource=parts[3], gain=float(parts[4]))

        elif type_char == 'H':
            if len(parts) < 5:
                return None
            return Element(name, 'CCV', n1=int(parts[1]), n2=int(parts[2]),
                          value=0.0, ctrl_vsource=parts[3], gain=float(parts[4]))

        return None


class NewtonRaphsonResult:
    def __init__(self):
        self.converged = False
        self.iterations = 0
        self.max_iterations = 0
        self.tolerance = 0.0
        self.residuals: List[float] = []
        self.iterate_history: List[np.ndarray] = []
        self.solution: Optional[np.ndarray] = None
        self.jacobian: Optional[np.ndarray] = None

    def print_debug(self, node_list: List[int], vsources: List[Element],
                    inductors: List[Element]):
        num_nodes = len(node_list)
        print("\n" + "=" * 70)
        print("NEWTON-RAPHSON DEBUG - DID NOT CONVERGE")
        print("=" * 70)
        print(f"\nMax iterations: {self.max_iterations}")
        print(f"Tolerance:      {self.tolerance:.2e}")
        print(f"Final residual: {self.residuals[-1]:.6e}")
        print(f"\nResidual history:")
        print(f"{'Iter':<6} {'||f(x)||':<20} {'||dx||':<20}")
        print("-" * 50)
        for i, res in enumerate(self.residuals):
            dx_norm = 0.0
            if i > 0 and len(self.iterate_history) > i:
                dx = self.iterate_history[i] - self.iterate_history[i - 1]
                dx_norm = np.linalg.norm(dx)
            print(f"{i:<6} {res:<20.6e} {dx_norm:<20.6e}")
        print(f"\nFinal iterate values:")
        print(f"{'Variable':<15} {'Real':<18} {'Imag':<18} {'Magnitude':<18}")
        print("-" * 70)
        for idx, node in enumerate(node_list):
            val = self.solution[idx]
            print(f"V({node}){'':<10} {val.real:<18.6e} {val.imag:<18.6e} {np.abs(val):<18.6e}")
        offset = num_nodes
        for idx, vs in enumerate(vsources):
            vs_idx = offset + idx
            val = self.solution[vs_idx]
            print(f"I({vs.name}){'':<9} {val.real:<18.6e} {val.imag:<18.6e} {np.abs(val):<18.6e}")
        offset += len(vsources)
        for idx, L in enumerate(inductors):
            L_idx = offset + idx
            val = self.solution[L_idx]
            print(f"I({L.name}){'':<9} {val.real:<18.6e} {val.imag:<18.6e} {np.abs(val):<18.6e}")
        if self.jacobian is not None:
            print(f"\nFinal Jacobian matrix ({self.jacobian.shape[0]} x {self.jacobian.shape[1]}):")
            for i in range(self.jacobian.shape[0]):
                row_str = "  ["
                for j in range(self.jacobian.shape[1]):
                    val = self.jacobian[i, j]
                    if abs(val.real) < 1e-10 and abs(val.imag) < 1e-10:
                        row_str += f"{'0':>12} "
                    else:
                        row_str += f"{val.real:>6.2f}{val.imag:>+6.2f}j "
                row_str += "]"
                print(row_str)
        print("=" * 70 + "\n")


class TransientResult:
    def __init__(self, time_points: np.ndarray, node_list: List[int],
                 element_names: List[str]):
        self.time = time_points
        self.node_list = node_list
        self.element_names = element_names
        self.node_voltages: Dict[int, np.ndarray] = {n: np.zeros_like(time_points) for n in [0] + node_list}
        self.branch_currents: Dict[str, np.ndarray] = {name: np.zeros_like(time_points) for name in element_names}

    def save_csv(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            header = ['time']
            for n in self.node_list:
                header.append(f"V({n})")
            for name in self.element_names:
                header.append(f"I({name})")
            f.write(','.join(header) + '\n')
            for i, t in enumerate(self.time):
                row = [f"{t:.6e}"]
                for n in self.node_list:
                    row.append(f"{self.node_voltages[n][i]:.6e}")
                for name in self.element_names:
                    row.append(f"{self.branch_currents[name][i]:.6e}")
                f.write(','.join(row) + '\n')
        print(f"\nTransient results saved to: {filename}")

    def plot(self, filename: str, nodes: Optional[List[int]] = None):
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping plot generation")
            return
        if nodes is None:
            nodes = self.node_list
        fig, ax = plt.subplots(figsize=(10, 6))
        for n in nodes:
            ax.plot(self.time, self.node_voltages[n], label=f'V({n})', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title('Transient Analysis - Node Voltages')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Waveform plot saved to: {filename}")


class CircuitSimulator:
    def __init__(self, parser: CircuitParser):
        self.parser = parser
        self.elements = parser.elements
        self.node_list: List[int] = []
        self.node_index: Dict[int, int] = {}
        self.voltage_sources: List[Element] = []
        self.vsource_index: Dict[str, int] = {}
        self.inductors: List[Element] = []
        self.inductor_index: Dict[str, int] = {}
        self.matrix_size: int = 0
        self.Y_matrix: Optional[np.ndarray] = None
        self.I_vector: Optional[np.ndarray] = None
        self.V_solution: Optional[np.ndarray] = None
        self.newton_result: Optional[NewtonRaphsonResult] = None
        self.has_controlled_sources = any(e.type in CONTROLLED_TYPES for e in self.elements)
        self.has_inductors = any(e.type == 'L' for e in self.elements)

    def _setup_nodes(self, for_transient: bool = False):
        non_ground_nodes = sorted([n for n in self.parser.nodes if n != 0])
        self.node_list = non_ground_nodes
        for i, node in enumerate(non_ground_nodes):
            self.node_index[node] = i

        self.voltage_sources = [e for e in self.elements if e.type in ('V', 'VCV', 'CCV')]
        for i, vs in enumerate(self.voltage_sources):
            self.vsource_index[vs.name] = i

        self.inductors = [e for e in self.elements if e.type == 'L'] if for_transient else []
        for i, L in enumerate(self.inductors):
            self.inductor_index[L.name] = i

        num_nodes = len(non_ground_nodes)
        num_vsources = len(self.voltage_sources)
        num_inductors = len(self.inductors)
        self.matrix_size = num_nodes + num_vsources + num_inductors

    def _get_admittance(self, element: Element, s: complex) -> complex:
        if element.type == 'R':
            return 1.0 / element.value
        elif element.type == 'C':
            return s * element.value
        elif element.type == 'L':
            return 1.0 / (s * element.value)
        return 0.0

    def _node_voltage_from_vec(self, x: np.ndarray, node: int) -> float:
        if node == 0:
            return 0.0
        idx = self.node_index.get(node)
        if idx is None:
            raise ValueError(f"Node {node} not found")
        return float(x[idx].real) if np.iscomplexobj(x) else float(x[idx])

    def _vsource_current_from_vec(self, x: np.ndarray, vsource_name: str) -> float:
        num_nodes = len(self.node_list)
        idx = self.vsource_index.get(vsource_name)
        if idx is None:
            raise ValueError(f"Voltage source {vsource_name} not found")
        val = x[num_nodes + idx]
        return float(val.real) if np.iscomplexobj(val) else float(val)

    def _inductor_current_from_vec(self, x: np.ndarray, inductor_name: str) -> float:
        num_nodes = len(self.node_list)
        num_vs = len(self.voltage_sources)
        idx = self.inductor_index.get(inductor_name)
        if idx is None:
            raise ValueError(f"Inductor {inductor_name} not found")
        val = x[num_nodes + num_vs + idx]
        return float(val.real) if np.iscomplexobj(val) else float(val)

    def build_mna_system(self, s: complex):
        self._setup_nodes(for_transient=False)
        n = self.matrix_size
        num_nodes = len(self.node_list)
        Y = np.zeros((n, n), dtype=complex)
        I = np.zeros(n, dtype=complex)

        for elem in self.elements:
            if elem.type == 'V' or elem.type in CONTROLLED_TYPES or elem.type == 'L':
                continue
            y = self._get_admittance(elem, s)
            ni, nj = elem.n1, elem.n2
            if ni != 0 and nj != 0:
                i, j = self.node_index[ni], self.node_index[nj]
                Y[i, i] += y; Y[j, j] += y; Y[i, j] -= y; Y[j, i] -= y
            elif ni != 0:
                i = self.node_index[ni]; Y[i, i] += y
            elif nj != 0:
                j = self.node_index[nj]; Y[j, j] += y

        for k, vs in enumerate(self.voltage_sources):
            if vs.type != 'V':
                continue
            col = num_nodes + k
            if vs.n1 != 0:
                i = self.node_index[vs.n1]; Y[i, col] += 1.0; Y[col, i] += 1.0
            if vs.n2 != 0:
                j = self.node_index[vs.n2]; Y[j, col] -= 1.0; Y[col, j] -= 1.0
            phase_rad = np.deg2rad(vs.ac_phase)
            I[col] = vs.ac_mag * np.exp(1j * phase_rad)

        self._add_controlled_sources_to_matrix(Y, I, s, num_nodes)

        self.Y_matrix = Y
        self.I_vector = I

    def _add_controlled_sources_to_matrix(self, Y: np.ndarray, I: np.ndarray,
                                           s: complex, num_nodes: int):
        for elem in self.elements:
            if elem.type not in CONTROLLED_TYPES:
                continue
            gain = elem.gain if elem.gain is not None else 1.0
            if elem.type == 'VCC':
                self._add_vcc_to_matrix(Y, elem, gain)
            elif elem.type == 'VCV':
                self._add_vcv_to_matrix(Y, I, elem, gain, num_nodes)
            elif elem.type == 'CCC':
                self._add_ccc_to_matrix(Y, elem, gain, num_nodes)
            elif elem.type == 'CCV':
                self._add_ccv_to_matrix(Y, I, elem, gain, num_nodes)

    def _add_vcc_to_matrix(self, Y: np.ndarray, elem: Element, gain: float):
        cn1, cn2, n1, n2 = elem.ctrl_n1, elem.ctrl_n2, elem.n1, elem.n2
        if cn1 != 0:
            ci = self.node_index[cn1]
            if n1 != 0: Y[self.node_index[n1], ci] += gain
            if n2 != 0: Y[self.node_index[n2], ci] -= gain
        if cn2 != 0:
            cj = self.node_index[cn2]
            if n1 != 0: Y[self.node_index[n1], cj] -= gain
            if n2 != 0: Y[self.node_index[n2], cj] += gain

    def _add_vcv_to_matrix(self, Y: np.ndarray, I: np.ndarray, elem: Element,
                           gain: float, num_nodes: int):
        vs_col = self.vsource_index[elem.name]
        col = num_nodes + vs_col
        cn1, cn2, n1, n2 = elem.ctrl_n1, elem.ctrl_n2, elem.n1, elem.n2
        if n1 != 0: i = self.node_index[n1]; Y[i, col] += 1.0; Y[col, i] += 1.0
        if n2 != 0: j = self.node_index[n2]; Y[j, col] -= 1.0; Y[col, j] -= 1.0
        if cn1 != 0: Y[col, self.node_index[cn1]] -= gain
        if cn2 != 0: Y[col, self.node_index[cn2]] += gain

    def _add_ccc_to_matrix(self, Y: np.ndarray, elem: Element, gain: float, num_nodes: int):
        vs_name = elem.ctrl_vsource
        if vs_name not in self.vsource_index:
            raise ValueError(f"Control voltage source '{vs_name}' not found for CCC {elem.name}")
        ctrl_col = num_nodes + self.vsource_index[vs_name]
        if elem.n1 != 0: Y[self.node_index[elem.n1], ctrl_col] += gain
        if elem.n2 != 0: Y[self.node_index[elem.n2], ctrl_col] -= gain

    def _add_ccv_to_matrix(self, Y: np.ndarray, I: np.ndarray, elem: Element,
                           gain: float, num_nodes: int):
        vs_name = elem.ctrl_vsource
        if vs_name not in self.vsource_index:
            raise ValueError(f"Control voltage source '{vs_name}' not found for CCV {elem.name}")
        ctrl_col = num_nodes + self.vsource_index[vs_name]
        elem_col = self.vsource_index[elem.name]
        col = num_nodes + elem_col
        if elem.n1 != 0: i = self.node_index[elem.n1]; Y[i, col] += 1.0; Y[col, i] += 1.0
        if elem.n2 != 0: j = self.node_index[elem.n2]; Y[j, col] -= 1.0; Y[col, j] -= 1.0
        Y[col, ctrl_col] -= gain

    def _solve_initial_condition(self, t0: float) -> np.ndarray:
        self._setup_nodes(for_transient=True)
        n = self.matrix_size
        num_nodes = len(self.node_list)
        num_vs = len(self.voltage_sources)

        G = np.zeros((n, n), dtype=float)
        I = np.zeros(n, dtype=float)

        for elem in self.elements:
            if elem.type == 'R':
                g = 1.0 / elem.value
                ni, nj = elem.n1, elem.n2
                if ni != 0 and nj != 0:
                    i, j = self.node_index[ni], self.node_index[nj]
                    G[i, i] += g; G[j, j] += g; G[i, j] -= g; G[j, i] -= g
                elif ni != 0: G[self.node_index[ni], self.node_index[ni]] += g
                elif nj != 0: G[self.node_index[nj], self.node_index[nj]] += g

            elif elem.type == 'C':
                ni, nj = elem.n1, elem.n2
                if ni != 0 and nj != 0:
                    i, j = self.node_index[ni], self.node_index[nj]
                    big_g = 1e12
                    G[i, i] += big_g; G[j, j] += big_g; G[i, j] -= big_g; G[j, i] -= big_g
                elif ni != 0:
                    i = self.node_index[ni]; G[i, i] += 1e12
                elif nj != 0:
                    j = self.node_index[nj]; G[j, j] += 1e12

            elif elem.type == 'L':
                n1, n2 = elem.n1, elem.n2
                L_idx = self.inductor_index[elem.name]
                col = num_nodes + num_vs + L_idx
                row = col
                if n1 != 0: G[self.node_index[n1], col] += 1.0
                if n2 != 0: G[self.node_index[n2], col] -= 1.0
                G[row, col] = 1.0
                I[row] = 0.0

        for k, vs in enumerate(self.voltage_sources):
            if vs.type != 'V':
                continue
            col = num_nodes + k
            v_src = vs.get_voltage(t0)
            if vs.n1 != 0: i = self.node_index[vs.n1]; G[i, col] += 1.0; G[col, i] += 1.0
            if vs.n2 != 0: j = self.node_index[vs.n2]; G[j, col] -= 1.0; G[col, j] -= 1.0
            I[col] = v_src

        for elem in self.elements:
            if elem.type not in CONTROLLED_TYPES:
                continue
            gain = elem.gain if elem.gain is not None else 1.0
            if elem.type == 'VCC':
                self._add_vcc_to_matrix(G, elem, gain)
            elif elem.type == 'VCV':
                self._add_vcv_to_matrix(G, I, elem, gain, num_nodes)
            elif elem.type == 'CCC':
                self._add_ccc_to_matrix(G, elem, gain, num_nodes)
            elif elem.type == 'CCV':
                self._add_ccv_to_matrix(G, I, elem, gain, num_nodes)

        return np.linalg.solve(G, I)

    def build_transient_system(self, t: float, dt: float,
                               prev_x: Optional[np.ndarray],
                               prev_vc: Dict[str, float],
                               prev_iL: Dict[str, float]):
        self._setup_nodes(for_transient=True)
        n = self.matrix_size
        num_nodes = len(self.node_list)
        num_vs = len(self.voltage_sources)

        G = np.zeros((n, n), dtype=float)
        I = np.zeros(n, dtype=float)

        for elem in self.elements:
            if elem.type == 'R':
                g = 1.0 / elem.value
                ni, nj = elem.n1, elem.n2
                if ni != 0 and nj != 0:
                    i, j = self.node_index[ni], self.node_index[nj]
                    G[i, i] += g; G[j, j] += g; G[i, j] -= g; G[j, i] -= g
                elif ni != 0: G[self.node_index[ni], self.node_index[ni]] += g
                elif nj != 0: G[self.node_index[nj], self.node_index[nj]] += g

            elif elem.type == 'C':
                c_eq = 2.0 * elem.value / dt
                ni, nj = elem.n1, elem.n2
                v_prev = 0.0
                if prev_x is not None:
                    v_prev = (self._node_voltage_from_vec(prev_x, ni)
                              - self._node_voltage_from_vec(prev_x, nj))
                i_prev = prev_vc.get(elem.name, 0.0)
                i_eq = c_eq * v_prev + i_prev
                if ni != 0 and nj != 0:
                    i, j = self.node_index[ni], self.node_index[nj]
                    G[i, i] += c_eq; G[j, j] += c_eq; G[i, j] -= c_eq; G[j, i] -= c_eq
                    I[i] += i_eq; I[j] -= i_eq
                elif ni != 0:
                    i = self.node_index[ni]; G[i, i] += c_eq; I[i] += i_eq
                elif nj != 0:
                    j = self.node_index[nj]; G[j, j] += c_eq; I[j] -= i_eq

            elif elem.type == 'L':
                l_eq = 2.0 * elem.value / dt
                n1, n2 = elem.n1, elem.n2
                L_idx = self.inductor_index[elem.name]
                col = num_nodes + num_vs + L_idx
                row = col
                v_prev = 0.0
                if prev_x is not None:
                    v_prev = (self._node_voltage_from_vec(prev_x, n1)
                              - self._node_voltage_from_vec(prev_x, n2))
                i_prev = prev_iL.get(elem.name, 0.0)
                if n1 != 0: G[self.node_index[n1], col] += 1.0
                if n2 != 0: G[self.node_index[n2], col] -= 1.0
                if n1 != 0: G[row, self.node_index[n1]] += 1.0
                if n2 != 0: G[row, self.node_index[n2]] -= 1.0
                G[row, col] -= l_eq
                I[row] -= l_eq * i_prev + v_prev

        for k, vs in enumerate(self.voltage_sources):
            if vs.type != 'V':
                continue
            col = num_nodes + k
            v_src = vs.get_voltage(t)
            if vs.n1 != 0: i = self.node_index[vs.n1]; G[i, col] += 1.0; G[col, i] += 1.0
            if vs.n2 != 0: j = self.node_index[vs.n2]; G[j, col] -= 1.0; G[col, j] -= 1.0
            I[col] = v_src

        for elem in self.elements:
            if elem.type not in CONTROLLED_TYPES:
                continue
            gain = elem.gain if elem.gain is not None else 1.0
            if elem.type == 'VCC':
                self._add_vcc_to_matrix(G, elem, gain)
            elif elem.type == 'VCV':
                self._add_vcv_to_matrix(G, I, elem, gain, num_nodes)
            elif elem.type == 'CCC':
                self._add_ccc_to_matrix(G, elem, gain, num_nodes)
            elif elem.type == 'CCV':
                self._add_ccv_to_matrix(G, I, elem, gain, num_nodes)

        return G, I

    def compute_residual(self, x: np.ndarray, G: np.ndarray, I: np.ndarray) -> np.ndarray:
        return G @ x - I

    def compute_jacobian(self, x: np.ndarray, G: np.ndarray) -> np.ndarray:
        return G.copy()

    def solve_newton_raphson(self, G: np.ndarray, I: np.ndarray,
                             max_iter: int = 50, tol: float = 1e-6,
                             x0: Optional[np.ndarray] = None,
                             is_complex: bool = False) -> NewtonRaphsonResult:
        result = NewtonRaphsonResult()
        result.max_iterations = max_iter
        result.tolerance = tol
        n = G.shape[0]
        dtype = complex if is_complex else float
        x = x0.copy() if x0 is not None else np.zeros(n, dtype=dtype)

        result.iterate_history.append(x.copy())
        residual = self.compute_residual(x, G, I)
        res_norm = np.linalg.norm(residual)
        result.residuals.append(res_norm)

        for iteration in range(max_iter):
            result.iterations = iteration + 1
            if res_norm < tol:
                result.converged = True
                result.solution = x
                result.jacobian = self.compute_jacobian(x, G)
                return result
            J = self.compute_jacobian(x, G)
            f = -residual
            try:
                dx = np.linalg.solve(J, f)
            except np.linalg.LinAlgError:
                dx = np.linalg.pinv(J) @ f
            x = x + dx
            result.iterate_history.append(x.copy())
            residual = self.compute_residual(x, G, I)
            res_norm = np.linalg.norm(residual)
            result.residuals.append(res_norm)
            dx_norm = np.linalg.norm(dx)
            if dx_norm < tol and res_norm < tol:
                result.converged = True
                result.solution = x
                result.jacobian = J
                return result

        result.converged = False
        result.solution = x
        result.jacobian = self.compute_jacobian(x, G)
        return result

    def solve(self, use_newton: Optional[bool] = None) -> np.ndarray:
        if self.Y_matrix is None or self.I_vector is None:
            raise ValueError("Matrix not built yet")
        max_iter = self.parser.newton_params['max_iter']
        tol = self.parser.newton_params['tol']
        if use_newton is None:
            use_newton = self.has_controlled_sources
        if use_newton:
            result = self.solve_newton_raphson(self.Y_matrix, self.I_vector,
                                               max_iter, tol, is_complex=True)
            self.newton_result = result
            if not result.converged:
                result.print_debug(self.node_list,
                                   [e for e in self.elements if e.type == 'V'],
                                   [])
                raise RuntimeError(
                    f"Newton-Raphson did not converge after {result.iterations} iterations. "
                    f"Final residual: {result.residuals[-1]:.6e}, tolerance: {tol:.2e}"
                )
            print(f"\nNewton-Raphson converged in {result.iterations} iterations "
                  f"(final residual: {result.residuals[-1]:.2e})")
            self.V_solution = result.solution
            return self.V_solution
        else:
            try:
                self.V_solution = np.linalg.solve(self.Y_matrix, self.I_vector)
            except np.linalg.LinAlgError:
                self.V_solution = np.linalg.pinv(self.Y_matrix) @ self.I_vector
            return self.V_solution

    def solve_transient_step(self, t: float, dt: float,
                              prev_x: Optional[np.ndarray],
                              prev_vc: Dict[str, float],
                              prev_iL: Dict[str, float],
                              x0: Optional[np.ndarray] = None) -> np.ndarray:
        G, I = self.build_transient_system(t, dt, prev_x, prev_vc, prev_iL)
        max_iter = self.parser.newton_params['max_iter']
        tol = self.parser.newton_params['tol']
        result = self.solve_newton_raphson(G, I, max_iter, tol, x0=x0, is_complex=False)
        if not result.converged:
            result.print_debug(self.node_list, self.voltage_sources, self.inductors)
            raise RuntimeError(
                f"Newton-Raphson did not converge at t={t:.6e}s. "
                f"Final residual: {result.residuals[-1]:.6e}, tolerance: {tol:.2e}"
            )
        return result.solution

    def simulate_transient(self) -> TransientResult:
        params = self.parser.tran_params
        tstep = params['tstep']
        tstop = params['tstop']
        tstart = params.get('tstart', 0.0)

        time_points = np.arange(tstart, tstop + tstep / 2, tstep)
        n_steps = len(time_points)

        self._setup_nodes(for_transient=True)

        result = TransientResult(time_points, self.node_list,
                                 [e.name for e in self.elements])

        prev_x = self._solve_initial_condition(time_points[0])
        prev_vc: Dict[str, float] = {}
        prev_iL: Dict[str, float] = {}

        x = prev_x.copy()
        for n in self.node_list:
            result.node_voltages[n][0] = self._node_voltage_from_vec(x, n)
        for elem in self.elements:
            I_branch = self.calc_branch_current_transient(elem, x, tstep, prev_vc, prev_iL)
            result.branch_currents[elem.name][0] = I_branch
        for elem in self.elements:
            if elem.type == 'C':
                prev_vc[elem.name] = 0.0
            elif elem.type == 'L':
                prev_iL[elem.name] = self._inductor_current_from_vec(x, elem.name)

        x0 = x.copy()
        print(f"  Transient: t={time_points[0]:.6e}s ({1/n_steps*100:5.1f}%)", end='\r')

        for i in range(1, len(time_points)):
            t = time_points[i]
            x = self.solve_transient_step(t, tstep, prev_x, prev_vc, prev_iL, x0)

            for n in self.node_list:
                result.node_voltages[n][i] = self._node_voltage_from_vec(x, n)

            for elem in self.elements:
                I_branch = self.calc_branch_current_transient(elem, x, tstep, prev_vc, prev_iL)
                result.branch_currents[elem.name][i] = I_branch

            for elem in self.elements:
                if elem.type == 'C':
                    v1 = self._node_voltage_from_vec(x, elem.n1)
                    v2 = self._node_voltage_from_vec(x, elem.n2)
                    v_new = v1 - v2
                    v_prev = (self._node_voltage_from_vec(prev_x, elem.n1)
                              - self._node_voltage_from_vec(prev_x, elem.n2))
                    i_prev = prev_vc.get(elem.name, 0.0)
                    prev_vc[elem.name] = (2 * elem.value / tstep) * (v_new - v_prev) - i_prev
                elif elem.type == 'L':
                    prev_iL[elem.name] = self._inductor_current_from_vec(x, elem.name)

            prev_x = x
            x0 = x.copy()

            if i % max(1, n_steps // 20) == 0 or i == n_steps - 1:
                print(f"  Transient: t={t:.6e}s ({(i+1)/n_steps*100:5.1f}%)", end='\r')

        print(f"\nTransient simulation complete: {n_steps} time steps")
        return result

    def get_node_voltage(self, node: int) -> complex:
        if node == 0:
            return 0.0 + 0.0j
        if self.V_solution is None:
            raise ValueError("System not solved yet")
        idx = self.node_index.get(node)
        if idx is None:
            raise ValueError(f"Node {node} not found")
        return self.V_solution[idx]

    def get_vsource_current(self, vsource_name: str) -> complex:
        if self.V_solution is None:
            raise ValueError("System not solved yet")
        num_nodes = len(self.node_list)
        idx = self.vsource_index.get(vsource_name)
        if idx is None:
            raise ValueError(f"Voltage source {vsource_name} not found")
        return self.V_solution[num_nodes + idx]

    def calc_branch_current(self, elem: Element, s: complex) -> complex:
        v1 = self.get_node_voltage(elem.n1)
        v2 = self.get_node_voltage(elem.n2)
        v_diff = v1 - v2
        if elem.type in ('V', 'VCV', 'CCV'):
            return self.get_vsource_current(elem.name)
        elif elem.type == 'VCC':
            ctrl_v = (self.get_node_voltage(elem.ctrl_n1)
                      - self.get_node_voltage(elem.ctrl_n2))
            gain = elem.gain if elem.gain is not None else 1.0
            return gain * ctrl_v
        elif elem.type == 'CCC':
            ctrl_i = self.get_vsource_current(elem.ctrl_vsource)
            gain = elem.gain if elem.gain is not None else 1.0
            return gain * ctrl_i
        else:
            y = self._get_admittance(elem, s)
            return y * v_diff

    def calc_branch_current_transient(self, elem: Element, x: np.ndarray,
                                       dt: float, prev_vc: Dict[str, float],
                                       prev_iL: Dict[str, float]) -> float:
        v1 = self._node_voltage_from_vec(x, elem.n1)
        v2 = self._node_voltage_from_vec(x, elem.n2)
        v_diff = v1 - v2
        if elem.type == 'V':
            return self._vsource_current_from_vec(x, elem.name)
        elif elem.type == 'R':
            return v_diff / elem.value
        elif elem.type == 'C':
            c_eq = 2.0 * elem.value / dt
            v_prev = 0.0
            i_prev = prev_vc.get(elem.name, 0.0)
            return c_eq * v_diff - (c_eq * v_prev + i_prev)
        elif elem.type == 'L':
            return self._inductor_current_from_vec(x, elem.name)
        elif elem.type == 'VCC':
            ctrl_v = (self._node_voltage_from_vec(x, elem.ctrl_n1)
                      - self._node_voltage_from_vec(x, elem.ctrl_n2))
            gain = elem.gain if elem.gain is not None else 1.0
            return gain * ctrl_v
        elif elem.type == 'VCV':
            return self._vsource_current_from_vec(x, elem.name)
        elif elem.type == 'CCC':
            ctrl_i = self._vsource_current_from_vec(x, elem.ctrl_vsource)
            gain = elem.gain if elem.gain is not None else 1.0
            return gain * ctrl_i
        elif elem.type == 'CCV':
            return self._vsource_current_from_vec(x, elem.name)
        return 0.0

    def generate_report(self, s: complex, output_file: str):
        num_nodes = len(self.node_list)
        num_vs = len([e for e in self.elements if e.type == 'V'])
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("CIRCUIT SIMULATION REPORT - AC Small Signal Analysis\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Circuit: {self.parser.title}\n\n")
            sigma, omega = s.real, s.imag
            freq = omega / (2 * np.pi) if omega != 0 else 0
            f.write(f"Complex frequency s = {sigma:.6f} + j{omega:.6f} rad/s\n")
            f.write(f"  sigma = {sigma:.6f} Np/s\n  omega = {omega:.6f} rad/s\n  f     = {freq:.6f} Hz\n\n")
            if self.newton_result is not None:
                f.write("Newton-Raphson Solver:\n")
                f.write(f"  Iterations: {self.newton_result.iterations}\n")
                f.write(f"  Converged:  {self.newton_result.converged}\n")
                f.write(f"  Final res:  {self.newton_result.residuals[-1]:.6e}\n")
                f.write(f"  Tolerance:  {self.newton_result.tolerance:.2e}\n\n")
            f.write("-" * 70 + "\nNODE VOLTAGES\n" + "-" * 70 + "\n")
            f.write(f"{'Node':<8} {'Magnitude (V)':<18} {'Phase (deg)':<15} {'Real':<15} {'Imag':<15}\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'0 (gnd)':<8} {0.0:<18.6e} {0.0:<15.6f} {0.0:<15.6e} {0.0:<15.6e}\n")
            for node in self.node_list:
                v = self.get_node_voltage(node)
                mag, phase = np.abs(v), np.angle(v, deg=True)
                f.write(f"{node:<8} {mag:<18.6e} {phase:<15.6f} {v.real:<15.6e} {v.imag:<15.6e}\n")
            f.write("\n" + "-" * 70 + "\nBRANCH CURRENTS\n" + "-" * 70 + "\n")
            f.write(f"{'Element':<12} {'Type':<6} {'Magnitude (A)':<18} {'Phase (deg)':<15} {'Real':<15} {'Imag':<15}\n")
            f.write("-" * 70 + "\n")
            for elem in self.elements:
                I_branch = self.calc_branch_current(elem, s)
                mag, phase = np.abs(I_branch), np.angle(I_branch, deg=True)
                f.write(f"{elem.name:<12} {elem.type:<6} {mag:<18.6e} {phase:<15.6f} {I_branch.real:<15.6e} {I_branch.imag:<15.6e}\n")
            f.write("\n" + "-" * 70 + "\nMATRIX DIMENSIONS\n" + "-" * 70 + "\n")
            f.write(f"Number of nodes (excl. ground): {num_nodes}\n")
            f.write(f"Number of voltage sources:      {num_vs}\n")
            f.write(f"Has dependent sources:          {self.has_controlled_sources}\n")
            f.write(f"MNA matrix size:                {self.matrix_size} x {self.matrix_size}\n")
            f.write("=" * 70 + "\n")

    def print_results(self, s: complex):
        sigma, omega = s.real, s.imag
        freq = omega / (2 * np.pi) if omega != 0 else 0
        print("\n" + "=" * 70)
        print("AC SMALL SIGNAL ANALYSIS RESULTS")
        print("=" * 70)
        print(f"\nComplex frequency s = {sigma:.6g} + j{omega:.6g} rad/s\n  (f = {freq:.6g} Hz)\n")
        print("-" * 70)
        print(f"{'Node':<8} {'|V| (V)':<16} {'Phase (deg)':<14} {'Re(V)':<14} {'Im(V)':<14}")
        print("-" * 70)
        print(f"{'0 (gnd)':<8} {0.0:<16.6e} {0.0:<14.6f} {0.0:<14.6e} {0.0:<14.6e}")
        for node in self.node_list:
            v = self.get_node_voltage(node)
            mag, phase = np.abs(v), np.angle(v, deg=True)
            print(f"{node:<8} {mag:<16.6e} {phase:<14.6f} {v.real:<14.6e} {v.imag:<14.6e}")
        print("-" * 70)


def run_simulation(input_file: str, output_file: str,
                   s: Optional[complex] = None,
                   use_newton: Optional[bool] = None,
                   csv_file: Optional[str] = None,
                   plot_file: Optional[str] = None):
    parser = CircuitParser(input_file)
    parser.parse()

    if not parser.elements:
        print("Error: No elements found in the circuit file.")
        return

    sim = CircuitSimulator(parser)

    if parser.analysis_mode == 'tran':
        print("\n" + "=" * 70)
        print("TRANSIENT ANALYSIS")
        print("=" * 70)
        params = parser.tran_params
        print(f"\nTime step: {params['tstep']:.6e} s")
        print(f"Stop time: {params['tstop']:.6e} s")
        print(f"Start time: {params.get('tstart', 0.0):.6e} s\n")

        result = sim.simulate_transient()

        if csv_file is None:
            csv_file = output_file.replace('.txt', '.csv').replace('.', '_tran.')
            if csv_file == output_file:
                csv_file = 'transient_results.csv'
        result.save_csv(csv_file)

        if plot_file is None:
            plot_file = csv_file.replace('.csv', '.png')
        result.plot(plot_file)

        print("\n" + "=" * 70)
        print("TRANSIENT ANALYSIS COMPLETE")
        print("=" * 70)

    else:
        if s is None:
            if parser.ac_type == 'sig':
                s = complex(parser.ac_params.get('sigma', 0.0),
                            parser.ac_params.get('omega', 0.0))
            elif parser.ac_type == 'single':
                s = complex(0.0, 2 * np.pi * parser.ac_params.get('freq', 0.0))
            else:
                s = complex(0.0, 0.0)

        sim.build_mna_system(s)
        sim.solve(use_newton)
        sim.print_results(s)
        sim.generate_report(s, output_file)
        print(f"\nReport saved to: {output_file}")


def main():
    argparser = argparse.ArgumentParser(
        description="Circuit Simulator - AC Small Signal + Transient Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported elements:
  R<name> <n1> <n2> <resistance>           - Resistor
  C<name> <n1> <n2> <capacitance>          - Capacitor
  L<name> <n1> <n2> <inductance>           - Inductor
  V<name> <n+> <n-> [DC <val>] [AC <mag> <phase>]
     [STEP t0=<val> V0=<val> V1=<val>]     - Step source
     [PULSE t0=<val> t1=<val> V0=<val> V1=<val> period=<val>] - Pulse
     [SIN V0=<val> Vamp=<val> freq=<val> phase=<val>] - Sine

Dependent sources:
  G<name> <n+> <n-> <nc+> <nc-> <gain>     - VCC
  E<name> <n+> <n-> <nc+> <nc-> <gain>     - VCV
  F<name> <n+> <n-> <v_ctrl> <gain>        - CCC
  H<name> <n+> <n-> <v_ctrl> <gain>        - CCV

Analysis commands:
  .ac sig <sigma> <omega>                  - Single complex frequency
  .ac single <freq>                        - Single frequency
  .tran <tstep> <tstop> [tstart]           - Transient analysis

Node 0 is ground.
        """
    )

    argparser.add_argument('input', help='Input circuit file')
    argparser.add_argument('-o', '--output', default='circuit_report.txt',
                           help='Output report file (AC analysis)')
    argparser.add_argument('--csv', default=None,
                           help='CSV output file for transient results')
    argparser.add_argument('--plot', default=None,
                           help='PNG plot file for transient waveforms')
    argparser.add_argument('--sigma', type=float, default=None,
                           help='Real part of s (Np/s)')
    argparser.add_argument('--omega', type=float, default=None,
                           help='Imaginary part of s (rad/s)')
    argparser.add_argument('--freq', type=float, default=None,
                           help='Frequency in Hz (AC only)')
    argparser.add_argument('--newton', action='store_true', default=None,
                           help='Force Newton-Raphson solver')
    argparser.add_argument('--no-newton', action='store_true',
                           help='Force direct linear solver')

    args = argparser.parse_args()

    s = None
    if args.freq is not None:
        s = complex(0.0, 2 * np.pi * args.freq)
    elif args.sigma is not None or args.omega is not None:
        s = complex(args.sigma or 0.0, args.omega or 0.0)

    use_newton = None
    if args.newton: use_newton = True
    elif args.no_newton: use_newton = False

    run_simulation(args.input, args.output, s, use_newton, args.csv, args.plot)


if __name__ == '__main__':
    main()
