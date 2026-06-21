#!/usr/bin/env python3
"""
Circuit Simulator - AC Small Signal Analysis with Dependent Sources
Supports R, L, C, independent V sources, and 4 types of dependent sources
Uses Modified Nodal Analysis (MNA) with Newton-Raphson iteration
All operations in complex frequency domain s = sigma + j*omega
"""

import sys
import argparse
import numpy as np
from typing import List, Tuple, Dict, Optional


CONTROLLED_TYPES = {'VCC', 'VCV', 'CCC', 'CCV'}


class Element:
    def __init__(self, name: str, type_: str, n1: int, n2: int, value: float,
                 ac_mag: float = 0.0, ac_phase: float = 0.0,
                 ctrl_n1: Optional[int] = None, ctrl_n2: Optional[int] = None,
                 ctrl_vsource: Optional[str] = None,
                 gain: Optional[float] = None):
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

    def __repr__(self):
        if self.type in CONTROLLED_TYPES:
            return f"{self.name}({self.type}): n{self.n1}-n{self.n2}, gain={self.gain}"
        return f"{self.name}({self.type}): n{self.n1}-n{self.n2}, val={self.value}"


class CircuitParser:
    def __init__(self, filename: str):
        self.filename = filename
        self.elements: List[Element] = []
        self.nodes: set = set()
        self.ac_type: str = "none"
        self.ac_params: Dict = {}
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
            self.ac_params = {
                'sigma': float(parts[2]),
                'omega': float(parts[3])
            }
        elif ac_type == 'lin' and len(parts) >= 5:
            self.ac_type = 'lin'
            self.ac_params = {
                'npoints': int(parts[2]),
                'fstart': float(parts[3]),
                'fstop': float(parts[4])
            }
        elif ac_type == 'single' and len(parts) >= 3:
            self.ac_type = 'single'
            self.ac_params = {
                'freq': float(parts[2])
            }

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
            n1 = int(parts[1])
            n2 = int(parts[2])
            value = 0.0
            ac_mag = 0.0
            ac_phase = 0.0

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
                else:
                    if value == 0.0:
                        value = float(parts[i])
                    i += 1

            return Element(name, 'V', n1, n2, value, ac_mag, ac_phase)

        elif type_char == 'G':
            if len(parts) < 6:
                return None
            return Element(name, 'VCC',
                           n1=int(parts[1]), n2=int(parts[2]),
                           value=0.0,
                           ctrl_n1=int(parts[3]), ctrl_n2=int(parts[4]),
                           gain=float(parts[5]))

        elif type_char == 'E':
            if len(parts) < 6:
                return None
            return Element(name, 'VCV',
                           n1=int(parts[1]), n2=int(parts[2]),
                           value=0.0,
                           ctrl_n1=int(parts[3]), ctrl_n2=int(parts[4]),
                           gain=float(parts[5]))

        elif type_char == 'F':
            if len(parts) < 5:
                return None
            return Element(name, 'CCC',
                           n1=int(parts[1]), n2=int(parts[2]),
                           value=0.0,
                           ctrl_vsource=parts[3],
                           gain=float(parts[4]))

        elif type_char == 'H':
            if len(parts) < 5:
                return None
            return Element(name, 'CCV',
                           n1=int(parts[1]), n2=int(parts[2]),
                           value=0.0,
                           ctrl_vsource=parts[3],
                           gain=float(parts[4]))

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

    def print_debug(self, node_list: List[int], vsources: List[Element]):
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

        for idx, vs in enumerate(vsources):
            vs_idx = num_nodes + idx
            val = self.solution[vs_idx]
            print(f"I({vs.name}){'':<9} {val.real:<18.6e} {val.imag:<18.6e} {np.abs(val):<18.6e}")

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


class CircuitSimulator:
    def __init__(self, parser: CircuitParser):
        self.parser = parser
        self.elements = parser.elements
        self.node_list: List[int] = []
        self.node_index: Dict[int, int] = {}
        self.voltage_sources: List[Element] = []
        self.vsource_index: Dict[str, int] = {}
        self.matrix_size: int = 0
        self.Y_matrix: Optional[np.ndarray] = None
        self.I_vector: Optional[np.ndarray] = None
        self.V_solution: Optional[np.ndarray] = None
        self.newton_result: Optional[NewtonRaphsonResult] = None
        self.has_controlled_sources = any(e.type in CONTROLLED_TYPES for e in self.elements)

    def _setup_nodes(self):
        non_ground_nodes = sorted([n for n in self.parser.nodes if n != 0])
        self.node_list = non_ground_nodes
        for i, node in enumerate(non_ground_nodes):
            self.node_index[node] = i

        self.voltage_sources = [e for e in self.elements if e.type in ('V', 'VCV', 'CCV')]
        for i, vs in enumerate(self.voltage_sources):
            self.vsource_index[vs.name] = i

        num_nodes = len(non_ground_nodes)
        num_vsources = len(self.voltage_sources)
        self.matrix_size = num_nodes + num_vsources

    def _get_admittance(self, element: Element, s: complex) -> complex:
        if element.type == 'R':
            return 1.0 / element.value
        elif element.type == 'C':
            return s * element.value
        elif element.type == 'L':
            return 1.0 / (s * element.value)
        return 0.0

    def _node_voltage_from_vec(self, x: np.ndarray, node: int) -> complex:
        if node == 0:
            return 0.0 + 0.0j
        idx = self.node_index.get(node)
        if idx is None:
            raise ValueError(f"Node {node} not found")
        return x[idx]

    def _vsource_current_from_vec(self, x: np.ndarray, vsource_name: str) -> complex:
        num_nodes = len(self.node_list)
        idx = self.vsource_index.get(vsource_name)
        if idx is None:
            raise ValueError(f"Voltage source {vsource_name} not found")
        return x[num_nodes + idx]

    def build_mna_system(self, s: complex):
        self._setup_nodes()

        n = self.matrix_size
        num_nodes = len(self.node_list)
        num_vs = len(self.voltage_sources)

        Y = np.zeros((n, n), dtype=complex)
        I = np.zeros(n, dtype=complex)

        for elem in self.elements:
            if elem.type == 'V' or elem.type in CONTROLLED_TYPES:
                continue

            y = self._get_admittance(elem, s)
            ni = elem.n1
            nj = elem.n2

            if ni != 0 and nj != 0:
                i = self.node_index[ni]
                j = self.node_index[nj]
                Y[i, i] += y
                Y[j, j] += y
                Y[i, j] -= y
                Y[j, i] -= y
            elif ni != 0:
                i = self.node_index[ni]
                Y[i, i] += y
            elif nj != 0:
                j = self.node_index[nj]
                Y[j, j] += y

        for k, vs in enumerate(self.voltage_sources):
            if vs.type != 'V':
                continue

            col = num_nodes + k

            if vs.n1 != 0:
                i = self.node_index[vs.n1]
                Y[i, col] += 1.0
                Y[col, i] += 1.0

            if vs.n2 != 0:
                j = self.node_index[vs.n2]
                Y[j, col] -= 1.0
                Y[col, j] -= 1.0

            phase_rad = np.deg2rad(vs.ac_phase)
            vs_ac_value = vs.ac_mag * np.exp(1j * phase_rad)
            I[col] = vs_ac_value

        self._add_controlled_sources_to_matrix(Y, I, s)

        self.Y_matrix = Y
        self.I_vector = I

    def _add_controlled_sources_to_matrix(self, Y: np.ndarray, I: np.ndarray, s: complex):
        num_nodes = len(self.node_list)

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
        cn1 = elem.ctrl_n1
        cn2 = elem.ctrl_n2
        n1 = elem.n1
        n2 = elem.n2

        if cn1 != 0:
            ci = self.node_index[cn1]
            if n1 != 0:
                i = self.node_index[n1]
                Y[i, ci] += gain
            if n2 != 0:
                j = self.node_index[n2]
                Y[j, ci] -= gain

        if cn2 != 0:
            cj = self.node_index[cn2]
            if n1 != 0:
                i = self.node_index[n1]
                Y[i, cj] -= gain
            if n2 != 0:
                j = self.node_index[n2]
                Y[j, cj] += gain

    def _add_vcv_to_matrix(self, Y: np.ndarray, I: np.ndarray, elem: Element, gain: float, num_nodes: int):
        vs_col = self.vsource_index[elem.name]
        col = num_nodes + vs_col

        cn1 = elem.ctrl_n1
        cn2 = elem.ctrl_n2
        n1 = elem.n1
        n2 = elem.n2

        if n1 != 0:
            i = self.node_index[n1]
            Y[i, col] += 1.0
            Y[col, i] += 1.0
        if n2 != 0:
            j = self.node_index[n2]
            Y[j, col] -= 1.0
            Y[col, j] -= 1.0

        if cn1 != 0:
            ci = self.node_index[cn1]
            Y[col, ci] -= gain
        if cn2 != 0:
            cj = self.node_index[cn2]
            Y[col, cj] += gain

    def _add_ccc_to_matrix(self, Y: np.ndarray, elem: Element, gain: float, num_nodes: int):
        vs_name = elem.ctrl_vsource

        if vs_name not in self.vsource_index:
            raise ValueError(f"Control voltage source '{vs_name}' not found for CCC {elem.name}")

        vs_col = self.vsource_index[vs_name]
        ctrl_col = num_nodes + vs_col

        n1 = elem.n1
        n2 = elem.n2

        if n1 != 0:
            i = self.node_index[n1]
            Y[i, ctrl_col] += gain
        if n2 != 0:
            j = self.node_index[n2]
            Y[j, ctrl_col] -= gain

    def _add_ccv_to_matrix(self, Y: np.ndarray, I: np.ndarray, elem: Element, gain: float, num_nodes: int):
        vs_name = elem.ctrl_vsource

        if vs_name not in self.vsource_index:
            raise ValueError(f"Control voltage source '{vs_name}' not found for CCV {elem.name}")

        vs_col = self.vsource_index[vs_name]
        ctrl_col = num_nodes + vs_col

        elem_col = self.vsource_index[elem.name]
        col = num_nodes + elem_col

        n1 = elem.n1
        n2 = elem.n2

        if n1 != 0:
            i = self.node_index[n1]
            Y[i, col] += 1.0
            Y[col, i] += 1.0
        if n2 != 0:
            j = self.node_index[n2]
            Y[j, col] -= 1.0
            Y[col, j] -= 1.0

        Y[col, ctrl_col] -= gain

    def compute_residual(self, x: np.ndarray, s: complex) -> np.ndarray:
        if self.Y_matrix is None:
            raise ValueError("Matrix not built yet")
        return self.Y_matrix @ x - self.I_vector

    def compute_jacobian(self, x: np.ndarray, s: complex) -> np.ndarray:
        return self.Y_matrix.copy()

    def solve_newton_raphson(self, s: complex,
                             max_iter: int = 50,
                             tol: float = 1e-6) -> NewtonRaphsonResult:
        if self.Y_matrix is None or self.I_vector is None:
            raise ValueError("Matrix not built yet")

        result = NewtonRaphsonResult()
        result.max_iterations = max_iter
        result.tolerance = tol

        n = self.matrix_size
        x = np.zeros(n, dtype=complex)

        result.iterate_history.append(x.copy())
        residual = self.compute_residual(x, s)
        res_norm = np.linalg.norm(residual)
        result.residuals.append(res_norm)

        for iteration in range(max_iter):
            result.iterations = iteration + 1

            if res_norm < tol:
                result.converged = True
                result.solution = x
                result.jacobian = self.compute_jacobian(x, s)
                return result

            J = self.compute_jacobian(x, s)
            f = -residual

            try:
                dx = np.linalg.solve(J, f)
            except np.linalg.LinAlgError:
                J_pinv = np.linalg.pinv(J)
                dx = J_pinv @ f

            x = x + dx

            result.iterate_history.append(x.copy())
            residual = self.compute_residual(x, s)
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
        result.jacobian = self.compute_jacobian(x, s)
        return result

    def solve(self, use_newton: Optional[bool] = None) -> np.ndarray:
        if self.Y_matrix is None or self.I_vector is None:
            raise ValueError("Matrix not built yet")

        max_iter = self.parser.newton_params['max_iter']
        tol = self.parser.newton_params['tol']

        if use_newton is None:
            use_newton = self.has_controlled_sources

        if use_newton:
            s = 0j
            result = self.solve_newton_raphson(s, max_iter, tol)
            self.newton_result = result

            if not result.converged:
                result.print_debug(self.node_list,
                                   [e for e in self.elements if e.type == 'V'])
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

        if elem.type == 'V':
            return self.get_vsource_current(elem.name)

        elif elem.type == 'VCV':
            return self.get_vsource_current(elem.name)

        elif elem.type == 'CCV':
            return self.get_vsource_current(elem.name)

        elif elem.type == 'VCC':
            ctrl_v1 = self.get_node_voltage(elem.ctrl_n1)
            ctrl_v2 = self.get_node_voltage(elem.ctrl_n2)
            ctrl_v = ctrl_v1 - ctrl_v2
            gain = elem.gain if elem.gain is not None else 1.0
            return gain * ctrl_v

        elif elem.type == 'CCC':
            ctrl_i = self.get_vsource_current(elem.ctrl_vsource)
            gain = elem.gain if elem.gain is not None else 1.0
            return gain * ctrl_i

        else:
            y = self._get_admittance(elem, s)
            return y * v_diff

    def generate_report(self, s: complex, output_file: str):
        num_nodes = len(self.node_list)
        num_vs = len([e for e in self.elements if e.type == 'V'])

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("CIRCUIT SIMULATION REPORT - AC Small Signal Analysis\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Circuit: {self.parser.title}\n\n")

            sigma = s.real
            omega = s.imag
            freq = omega / (2 * np.pi) if omega != 0 else 0
            f.write(f"Complex frequency s = {sigma:.6f} + j{omega:.6f} rad/s\n")
            f.write(f"  sigma = {sigma:.6f} Np/s\n")
            f.write(f"  omega = {omega:.6f} rad/s\n")
            f.write(f"  f     = {freq:.6f} Hz\n\n")

            if self.newton_result is not None:
                f.write("Newton-Raphson Solver:\n")
                f.write(f"  Iterations: {self.newton_result.iterations}\n")
                f.write(f"  Converged:  {self.newton_result.converged}\n")
                f.write(f"  Final res:  {self.newton_result.residuals[-1]:.6e}\n")
                f.write(f"  Tolerance:  {self.newton_result.tolerance:.2e}\n\n")

            f.write("-" * 70 + "\n")
            f.write("NODE VOLTAGES\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Node':<8} {'Magnitude (V)':<18} {'Phase (deg)':<15} {'Real':<15} {'Imag':<15}\n")
            f.write("-" * 70 + "\n")

            f.write(f"{'0 (gnd)':<8} {0.0:<18.6e} {0.0:<15.6f} {0.0:<15.6e} {0.0:<15.6e}\n")

            for node in self.node_list:
                v = self.get_node_voltage(node)
                mag = np.abs(v)
                phase = np.angle(v, deg=True)
                f.write(f"{node:<8} {mag:<18.6e} {phase:<15.6f} {v.real:<15.6e} {v.imag:<15.6e}\n")

            f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("BRANCH CURRENTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Element':<12} {'Type':<6} {'Magnitude (A)':<18} {'Phase (deg)':<15} {'Real':<15} {'Imag':<15}\n")
            f.write("-" * 70 + "\n")

            for elem in self.elements:
                I_branch = self.calc_branch_current(elem, s)
                mag = np.abs(I_branch)
                phase = np.angle(I_branch, deg=True)
                f.write(f"{elem.name:<12} {elem.type:<6} {mag:<18.6e} {phase:<15.6f} {I_branch.real:<15.6e} {I_branch.imag:<15.6e}\n")

            f.write("\n")
            f.write("-" * 70 + "\n")
            f.write("MATRIX DIMENSIONS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Number of nodes (excl. ground): {num_nodes}\n")
            f.write(f"Number of voltage sources:      {num_vs}\n")
            f.write(f"Has dependent sources:          {self.has_controlled_sources}\n")
            f.write(f"MNA matrix size:                {self.matrix_size} x {self.matrix_size}\n")
            f.write("=" * 70 + "\n")

    def print_results(self, s: complex):
        sigma = s.real
        omega = s.imag
        freq = omega / (2 * np.pi) if omega != 0 else 0

        print("\n" + "=" * 70)
        print("AC SMALL SIGNAL ANALYSIS RESULTS")
        print("=" * 70)
        print(f"\nComplex frequency s = {sigma:.6g} + j{omega:.6g} rad/s")
        print(f"  (f = {freq:.6g} Hz)\n")

        print("-" * 70)
        print(f"{'Node':<8} {'|V| (V)':<16} {'Phase (deg)':<14} {'Re(V)':<14} {'Im(V)':<14}")
        print("-" * 70)

        print(f"{'0 (gnd)':<8} {0.0:<16.6e} {0.0:<14.6f} {0.0:<14.6e} {0.0:<14.6e}")

        for node in self.node_list:
            v = self.get_node_voltage(node)
            mag = np.abs(v)
            phase = np.angle(v, deg=True)
            print(f"{node:<8} {mag:<16.6e} {phase:<14.6f} {v.real:<14.6e} {v.imag:<14.6e}")

        print("-" * 70)


def run_simulation(input_file: str, output_file: str, s: Optional[complex] = None,
                   use_newton: Optional[bool] = None):
    parser = CircuitParser(input_file)
    parser.parse()

    if not parser.elements:
        print("Error: No elements found in the circuit file.")
        return

    sim = CircuitSimulator(parser)

    if s is None:
        if parser.ac_type == 'sig':
            sigma = parser.ac_params.get('sigma', 0.0)
            omega = parser.ac_params.get('omega', 0.0)
            s = complex(sigma, omega)
        elif parser.ac_type == 'single':
            freq = parser.ac_params.get('freq', 0.0)
            s = complex(0.0, 2 * np.pi * freq)
        else:
            s = complex(0.0, 0.0)

    sim.build_mna_system(s)
    sim.solve(use_newton)
    sim.print_results(s)
    sim.generate_report(s, output_file)

    print(f"\nReport saved to: {output_file}")


def main():
    argparser = argparse.ArgumentParser(
        description="Circuit Simulator - AC Small Signal Analysis with Dependent Sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported elements:
  R<name> <n1> <n2> <resistance>           - Resistor
  C<name> <n1> <n2> <capacitance>          - Capacitor
  L<name> <n1> <n2> <inductance>           - Inductor
  V<name> <n+> <n-> [DC <val>] [AC <mag> <phase>]  - Voltage source

Dependent sources:
  G<name> <n+> <n-> <nc+> <nc-> <gain>     - VCC (Voltage-Controlled Current Source)
  E<name> <n+> <n-> <nc+> <nc-> <gain>     - VCV (Voltage-Controlled Voltage Source)
  F<name> <n+> <n-> <v_ctrl> <gain>        - CCC (Current-Controlled Current Source)
  H<name> <n+> <n-> <v_ctrl> <gain>        - CCV (Current-Controlled Voltage Source)

Analysis commands:
  .ac sig <sigma> <omega>                  - Single complex frequency
  .ac single <freq>                        - Single frequency (sinusoidal)
  .ac lin <npts> <fstart> <fstop>          - Linear frequency sweep

Solver options:
  .options newton_maxiter <N>              - Newton max iterations (default: 50)
  .options newton_tol <tol>                - Newton tolerance (default: 1e-6)

Node 0 is ground.
        """
    )

    argparser.add_argument('input', help='Input circuit file (simplified SPICE format)')
    argparser.add_argument('-o', '--output', default='circuit_report.txt',
                           help='Output report file (default: circuit_report.txt)')
    argparser.add_argument('--sigma', type=float, default=None,
                           help='Real part of complex frequency s (sigma, Np/s)')
    argparser.add_argument('--omega', type=float, default=None,
                           help='Imaginary part of complex frequency s (omega, rad/s)')
    argparser.add_argument('--freq', type=float, default=None,
                           help='Frequency in Hz (sets omega = 2*pi*freq, sigma=0)')
    argparser.add_argument('--newton', action='store_true', default=None,
                           help='Force use of Newton-Raphson solver')
    argparser.add_argument('--no-newton', action='store_true',
                           help='Force use of direct linear solver (no Newton)')

    args = argparser.parse_args()

    s = None
    if args.freq is not None:
        s = complex(0.0, 2 * np.pi * args.freq)
    elif args.sigma is not None or args.omega is not None:
        sigma = args.sigma if args.sigma is not None else 0.0
        omega = args.omega if args.omega is not None else 0.0
        s = complex(sigma, omega)

    use_newton = None
    if args.newton:
        use_newton = True
    elif args.no_newton:
        use_newton = False

    run_simulation(args.input, args.output, s, use_newton)


if __name__ == '__main__':
    main()
