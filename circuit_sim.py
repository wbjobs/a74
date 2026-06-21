#!/usr/bin/env python3
"""
Circuit Simulator - AC Small Signal Analysis
Supports R, L, C, independent voltage sources
Uses Modified Nodal Analysis (MNA) in complex frequency domain
"""

import sys
import argparse
import numpy as np
from typing import List, Tuple, Dict, Optional


class Element:
    def __init__(self, name: str, type_: str, n1: int, n2: int, value: float,
                 ac_mag: float = 0.0, ac_phase: float = 0.0):
        self.name = name
        self.type = type_
        self.n1 = n1
        self.n2 = n2
        self.value = value
        self.ac_mag = ac_mag
        self.ac_phase = ac_phase

    def __repr__(self):
        return f"{self.name}({self.type}): n{self.n1}-n{self.n2}, val={self.value}"


class CircuitParser:
    def __init__(self, filename: str):
        self.filename = filename
        self.elements: List[Element] = []
        self.nodes: set = set()
        self.ac_type: str = "none"
        self.ac_params: Dict = {}
        self.title: str = ""

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

            element = self._parse_element(parts)
            if element:
                self.elements.append(element)
                self.nodes.add(element.n1)
                self.nodes.add(element.n2)

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

        return None


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

    def _setup_nodes(self):
        non_ground_nodes = sorted([n for n in self.parser.nodes if n != 0])
        self.node_list = non_ground_nodes
        for i, node in enumerate(non_ground_nodes):
            self.node_index[node] = i

        self.voltage_sources = [e for e in self.elements if e.type == 'V']
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

    def build_mna_matrix(self, s: complex):
        self._setup_nodes()

        n = self.matrix_size
        num_nodes = len(self.node_list)
        num_vs = len(self.voltage_sources)

        M = np.zeros((n, n), dtype=complex)
        I = np.zeros(n, dtype=complex)

        for elem in self.elements:
            if elem.type == 'V':
                continue

            y = self._get_admittance(elem, s)
            ni = elem.n1
            nj = elem.n2

            if ni != 0 and nj != 0:
                i = self.node_index[ni]
                j = self.node_index[nj]
                M[i, i] += y
                M[j, j] += y
                M[i, j] -= y
                M[j, i] -= y
            elif ni != 0:
                i = self.node_index[ni]
                M[i, i] += y
            elif nj != 0:
                j = self.node_index[nj]
                M[j, j] += y

        for k, vs in enumerate(self.voltage_sources):
            col = num_nodes + k

            if vs.n1 != 0:
                i = self.node_index[vs.n1]
                M[i, col] += 1.0
                M[col, i] += 1.0

            if vs.n2 != 0:
                j = self.node_index[vs.n2]
                M[j, col] -= 1.0
                M[col, j] -= 1.0

            phase_rad = np.deg2rad(vs.ac_phase)
            vs_ac_value = vs.ac_mag * np.exp(1j * phase_rad)
            I[col] = vs_ac_value

        self.Y_matrix = M
        self.I_vector = I

    def solve(self) -> np.ndarray:
        if self.Y_matrix is None or self.I_vector is None:
            raise ValueError("Matrix not built yet")

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
        else:
            y = self._get_admittance(elem, s)
            return y * v_diff

    def generate_report(self, s: complex, output_file: str):
        num_nodes = len(self.node_list)
        num_vs = len(self.voltage_sources)

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


def run_simulation(input_file: str, output_file: str, s: Optional[complex] = None):
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

    sim.build_mna_matrix(s)
    sim.solve()
    sim.print_results(s)
    sim.generate_report(s, output_file)

    print(f"\nReport saved to: {output_file}")


def main():
    argparser = argparse.ArgumentParser(
        description="Circuit Simulator - AC Small Signal Analysis (MNA-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported elements:
  R<name> <n1> <n2> <resistance>        - Resistor
  C<name> <n1> <n2> <capacitance>       - Capacitor
  L<name> <n1> <n2> <inductance>        - Inductor
  V<name> <n+> <n-> [DC <val>] [AC <mag> <phase>] - Voltage source

Analysis commands:
  .ac sig <sigma> <omega>               - Single complex frequency
  .ac single <freq>                     - Single frequency (sinusoidal)
  .ac lin <npts> <fstart> <fstop>       - Linear frequency sweep

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

    args = argparser.parse_args()

    s = None
    if args.freq is not None:
        s = complex(0.0, 2 * np.pi * args.freq)
    elif args.sigma is not None or args.omega is not None:
        sigma = args.sigma if args.sigma is not None else 0.0
        omega = args.omega if args.omega is not None else 0.0
        s = complex(sigma, omega)

    run_simulation(args.input, args.output, s)


if __name__ == '__main__':
    main()
