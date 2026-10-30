from __future__ import annotations

# Quantum_Putanje

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import time
import traceback

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import qiskit
import scipy

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, Statevector
from scipy.optimize import minimize
from scipy.special import logsumexp


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_2977.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_plus_1733.csv"),
)


NUMBERS = 39
PICKS = 7
QUBITS = 6
DIMENSION = 1 << QUBITS
PARAMETERS = 2 * DIMENSION

HISTORY_COEFFICIENTS = (1.0, 0.5, 0.25)
INPUT_PHASE_SCALE = 1.1

RIDGES = (0.03, 0.3)
MAX_ITERATIONS = 160
MAX_FUNCTION_CALLS = 1200
GRADIENT_LIMIT = 5.0e-5
CHECKPOINT_EVERY = 10

VERSION = "QUANTUM_PUTANJE_V1_20261029"

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CHECKPOINT_DIRECTORY = (
    SCRIPT_DIRECTORY / "Quantum_Putanje_V1_checkpoints"
)

PARAMETER_BOUNDS = [(-math.pi, math.pi)] * PARAMETERS

_path_indices = np.arange(DIMENSION, dtype=float)

BASE_PHASES_1 = (
    0.31 * np.sin(0.37 * (_path_indices + 1.0))
)

BASE_PHASES_2 = (
    0.29 * np.cos(0.41 * (_path_indices + 1.0))
)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def check_finite(values, label):
    require(
        np.all(np.isfinite(values)),
        f"{label}: pronađena je nekonačna ili nedefinisana vrednost.",
    )


def make_preparation():
    circuit = QuantumCircuit(QUBITS, name="Preparation")

    for qubit in range(QUBITS):
        circuit.ry(1.03 + 0.06 * qubit, qubit)
        circuit.rz(0.13 * (qubit + 1), qubit)

    for qubit in range(QUBITS - 1):
        circuit.cx(qubit, qubit + 1)

    return circuit


def make_mixer_1():
    circuit = QuantumCircuit(QUBITS, name="Mixer_1")

    for qubit in range(QUBITS):
        circuit.ry(1.13 + 0.07 * qubit, qubit)

    for qubit in range(QUBITS - 1):
        circuit.cx(qubit, qubit + 1)

    for qubit in range(QUBITS):
        circuit.rz(0.21 * (qubit + 1), qubit)

    return circuit


def make_mixer_2():
    circuit = QuantumCircuit(QUBITS, name="Mixer_2")

    for qubit in range(QUBITS):
        circuit.rx(0.80 + 0.05 * qubit, qubit)

    for qubit in range(QUBITS - 1, 0, -1):
        circuit.cx(qubit, qubit - 1)

    for qubit in range(QUBITS):
        circuit.ry(0.39 + 0.04 * qubit, qubit)

    return circuit


class QuantumPaths:
    def __init__(self):
        self.preparation = make_preparation()
        self.mixer_1 = make_mixer_1()
        self.mixer_2 = make_mixer_2()

        self.initial_amplitudes = np.asarray(
            Statevector.from_instruction(
                self.preparation
            ).data,
            dtype=np.complex128,
        )

        self.matrix_1 = np.asarray(
            Operator(self.mixer_1).data,
            dtype=np.complex128,
        )

        self.matrix_2 = np.asarray(
            Operator(self.mixer_2).data,
            dtype=np.complex128,
        )

        identity = np.eye(DIMENSION, dtype=np.complex128)

        for name, matrix in (
            ("Prvi mešač", self.matrix_1),
            ("Drugi mešač", self.matrix_2),
        ):
            error = np.max(
                np.abs(matrix.conj().T @ matrix - identity)
            )

            require(
                error < 1.0e-11,
                f"{name}: operator nije numerički unitaran.",
            )

        require(
            abs(
                float(
                    np.sum(
                        np.abs(self.initial_amplitudes) ** 2
                    )
                ) - 1.0
            ) < 1.0e-11,
            "Početno kvantno stanje nije normalizovano.",
        )

    @staticmethod
    def initial_parameters():
        return np.zeros(PARAMETERS, dtype=float)

    @staticmethod
    def phase_factors(parameters, zero_actions=False):
        parameters = np.asarray(parameters, dtype=float)

        require(
            parameters.shape == (PARAMETERS,),
            "Pogrešan broj faznih parametara.",
        )

        check_finite(parameters, "Fazni parametri")

        if zero_actions:
            return (
                np.ones(DIMENSION, dtype=np.complex128),
                np.ones(DIMENSION, dtype=np.complex128),
            )

        phase_1 = BASE_PHASES_1 + parameters[:DIMENSION]
        phase_2 = BASE_PHASES_2 + parameters[DIMENSION:]

        return (
            np.exp(1j * phase_1),
            np.exp(1j * phase_2),
        )

    def action_circuit(self, parameters, zero_actions=False):
        factor_1, factor_2 = self.phase_factors(
            parameters,
            zero_actions=zero_actions,
        )

        circuit = QuantumCircuit(QUBITS, name="Path_interference")

        circuit.unitary(
            np.diag(factor_1),
            list(range(QUBITS)),
            label="Action_1",
        )

        circuit.compose(self.mixer_1, inplace=True)

        circuit.unitary(
            np.diag(factor_2),
            list(range(QUBITS)),
            label="Action_2",
        )

        circuit.compose(self.mixer_2, inplace=True)

        return circuit

    def action_operator(self, parameters, zero_actions=False):
        circuit = self.action_circuit(
            parameters,
            zero_actions=zero_actions,
        )

        matrix = np.asarray(
            Operator(circuit).data,
            dtype=np.complex128,
        )

        check_finite(matrix, "Qiskit operator")

        return matrix

    @staticmethod
    def input_phases(features):
        features = np.asarray(features, dtype=float)

        require(
            features.ndim == 2
            and features.shape[1] == NUMBERS,
            "Pogrešan oblik istorijskog ulaza.",
        )

        check_finite(features, "Istorijski ulaz")

        phases = np.zeros(
            (len(features), DIMENSION),
            dtype=float,
        )

        phases[:, :NUMBERS] = (
            INPUT_PHASE_SCALE * features
        )

        return phases

    def prepare_inputs(self, features):
        phases = self.input_phases(features)

        states = (
            self.initial_amplitudes[None, :]
            * np.exp(1j * phases)
        )

        norms = np.sum(np.abs(states) ** 2, axis=1)

        require(
            np.max(np.abs(norms - 1.0)) < 1.0e-11,
            "Istorijska ulazna stanja nisu normalizovana.",
        )

        return np.ascontiguousarray(states)

    def local_circuit(
        self,
        parameters,
        features_row,
        zero_actions=False,
    ):
        phases = self.input_phases(
            np.asarray(features_row, dtype=float)[None, :]
        )[0]

        circuit = self.preparation.copy()

        circuit.unitary(
            np.diag(np.exp(1j * phases)),
            list(range(QUBITS)),
            label="History",
        )

        circuit.compose(
            self.action_circuit(
                parameters,
                zero_actions=zero_actions,
            ),
            inplace=True,
        )

        return circuit

    def seven_register_circuit(self, parameters, features_row):
        registers = [
            QuantumRegister(QUBITS, f"r{index + 1}")
            for index in range(PICKS)
        ]

        circuit = QuantumCircuit(*registers)
        local = self.local_circuit(parameters, features_row)

        for register in registers:
            circuit.compose(
                local,
                qubits=list(register),
                inplace=True,
            )

        require(
            circuit.num_qubits == PICKS * QUBITS,
            "Pogrešan broj kubita u sedam registara.",
        )

        return circuit

    def forward(
        self,
        parameters,
        inputs,
        derivatives=False,
        zero_actions=False,
    ):
        inputs = np.asarray(inputs, dtype=np.complex128)

        require(
            inputs.ndim == 2
            and inputs.shape[1] == DIMENSION
            and len(inputs) > 0,
            "Pogrešan oblik ulaznih amplituda.",
        )

        # Izlaz svih uzoraka nastaje primenom operatora
        # stvarnog Qiskit kola na pripremljena kvantna stanja.
        operator = self.action_operator(
            parameters,
            zero_actions=zero_actions,
        )

        output = inputs @ operator.T

        check_finite(output, "Izlazne amplitude")

        norms = np.sum(np.abs(output) ** 2, axis=1)

        require(
            np.max(np.abs(norms - 1.0)) < 2.0e-10,
            "Qiskit izlaz nije normalizovan.",
        )

        weights = np.abs(output[:, :NUMBERS]) ** 2

        require(
            np.all(weights > 0.0),
            "Kandidat je na granici domena: nulta amplituda broja.",
        )

        require(
            np.all(np.sum(weights, axis=1) <= 1.0 + 2.0e-10),
            "Masa dozvoljenih brojeva prelazi jedan.",
        )

        if not derivatives:
            return weights, output, None

        require(
            not zero_actions,
            "Gradijent nije definisan za kontrolu sa nuliranim fazama.",
        )

        factor_1, factor_2 = self.phase_factors(parameters)

        first = inputs * factor_1[None, :]
        middle = first @ self.matrix_1.T
        second = middle * factor_2[None, :]
        reconstructed = second @ self.matrix_2.T

        require(
            np.max(np.abs(reconstructed - output)) < 2.0e-10,
            "Razlaganje putanja se ne slaže sa Qiskit operatorom.",
        )

        intermediates = (
            first,
            second,
            factor_2,
        )

        return weights, output, intermediates

    def dephased_weights(self, inputs):
        # Potpuno uklanjanje koherencije između putanja.
        # Ovo je dijagnostička kontrola, bez dodatnog učenja.
        probabilities = np.abs(inputs) ** 2

        probabilities = (
            probabilities @ (np.abs(self.matrix_1) ** 2).T
        )

        probabilities = (
            probabilities @ (np.abs(self.matrix_2) ** 2).T
        )

        require(
            np.max(
                np.abs(
                    np.sum(probabilities, axis=1) - 1.0
                )
            ) < 2.0e-10,
            "Dekoherentna kontrola nije normalizovana.",
        )

        weights = probabilities[:, :NUMBERS]

        require(
            np.all(weights > 0.0),
            "Dekoherentna kontrola ima nultu težinu.",
        )

        return weights


def conditional_distribution(weights, picks=PICKS, marginals=False):
    """
    P(S) = product(weights[i], i in S) / e_picks(weights).

    Normalizacija obuhvata sve podskupove zadate veličine.
    Računanje koristi logaritamsku dinamičku rekurziju.
    """

    weights = np.asarray(weights, dtype=float)

    require(
        weights.ndim == 2 and len(weights) > 0,
        "Pogrešan oblik težina raspodele.",
    )

    samples, count = weights.shape

    require(
        0 <= picks <= count,
        "Neispravna veličina podskupa.",
    )

    require(
        np.all(np.isfinite(weights))
        and np.all(weights > 0.0),
        "Težine moraju biti konačne i strogo pozitivne.",
    )

    log_weights = np.log(weights)

    forward = np.full(
        (samples, count + 1, picks + 1),
        -np.inf,
        dtype=float,
    )

    forward[:, 0, 0] = 0.0

    for index in range(count):
        forward[:, index + 1, :] = forward[:, index, :]

        if picks > 0:
            forward[:, index + 1, 1:] = np.logaddexp(
                forward[:, index, 1:],
                log_weights[:, index, None]
                + forward[:, index, :picks],
            )

    log_normalizer = forward[:, count, picks]

    check_finite(log_normalizer, "Normalizaciona konstanta")

    if not marginals:
        return log_normalizer, None

    if picks == 0:
        return (
            log_normalizer,
            np.zeros_like(weights),
        )

    backward = np.full_like(forward, -np.inf)
    backward[:, count, 0] = 0.0

    for index in range(count - 1, -1, -1):
        backward[:, index, :] = backward[:, index + 1, :]

        backward[:, index, 1:] = np.logaddexp(
            backward[:, index + 1, 1:],
            log_weights[:, index, None]
            + backward[:, index + 1, :picks],
        )

    require(
        np.max(
            np.abs(
                backward[:, 0, picks] - log_normalizer
            )
        ) < 2.0e-10,
        "Prednja i zadnja normalizacija se ne slažu.",
    )

    inclusion = np.empty_like(weights)

    for index in range(count):
        rest = logsumexp(
            forward[:, index, :picks]
            + backward[
                :,
                index + 1,
                picks - 1::-1,
            ],
            axis=1,
        )

        inclusion[:, index] = np.exp(
            log_weights[:, index]
            + rest
            - log_normalizer
        )

    check_finite(inclusion, "Marginalne verovatnoće")

    require(
        np.all(inclusion >= 0.0)
        and np.all(inclusion <= 1.0 + 2.0e-8),
        "Marginalna verovatnoća je izvan dozvoljenog intervala.",
    )

    require(
        np.max(
            np.abs(np.sum(inclusion, axis=1) - picks)
        ) < 2.0e-7,
        "Zbir marginalnih verovatnoća nije jednak broju izbora.",
    )

    return log_normalizer, inclusion


def subset_log_probabilities(weights, targets, picks=PICKS):
    targets = np.asarray(targets, dtype=float)

    require(
        targets.shape == weights.shape,
        "Pogrešan oblik ciljnih kombinacija.",
    )

    require(
        np.all((targets == 0.0) | (targets == 1.0))
        and np.all(np.sum(targets, axis=1) == picks),
        "Ciljevi nisu legalni podskupovi.",
    )

    log_normalizer, _ = conditional_distribution(
        weights,
        picks=picks,
        marginals=False,
    )

    log_probability = (
        np.sum(targets * np.log(weights), axis=1)
        - log_normalizer
    )

    require(
        np.all(log_probability <= 2.0e-8),
        "Verovatnoća kombinacije prelazi jedan.",
    )

    check_finite(log_probability, "Verovatnoće ciljnih kombinacija")

    return log_probability


def exact_maximum(weights, picks=PICKS):
    weights = np.asarray(weights, dtype=float)

    require(
        weights.ndim == 1
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0),
        "Neispravne težine za izbor maksimuma.",
    )

    require(
        0 <= picks <= len(weights),
        "Neispravan broj izbora za maksimum.",
    )

    # Za ovu proizvodnu raspodelu ovo je tačan globalni MAP.
    # Rangiraju se naučene Qiskit težine, ne istorijske učestalosti.
    order = np.argsort(-weights, kind="stable")
    selected = np.sort(order[:picks])

    log_normalizer, _ = conditional_distribution(
        weights[None, :],
        picks=picks,
        marginals=False,
    )

    log_probability = (
        float(np.sum(np.log(weights[selected])))
        - float(log_normalizer[0])
    )

    lower_bound = -math.log(math.comb(len(weights), picks))

    require(
        lower_bound - 2.0e-8
        <= log_probability
        <= 2.0e-8,
        "Izabrani maksimum ne prolazi granice verovatnoće.",
    )

    return tuple(int(number) for number in selected), log_probability


def independent_normalizer_check(weights, picks=PICKS):
    weights = np.asarray(weights, dtype=np.longdouble)

    scale = np.max(weights)
    require(scale > 0.0, "Nulta skala normalizacije.")

    scaled = weights / scale
    coefficients = np.zeros(picks + 1, dtype=np.longdouble)
    coefficients[0] = 1.0

    for index, weight in enumerate(scaled):
        for degree in range(min(picks, index + 1), 0, -1):
            coefficients[degree] += (
                weight * coefficients[degree - 1]
            )

    require(
        coefficients[picks] > 0.0,
        "Alternativna normalizacija nije pozitivna.",
    )

    return float(
        np.log(coefficients[picks])
        + picks * np.log(scale)
    )


def objective(parameters, source, inputs, targets, ridge):
    weights, output, intermediates = source.forward(
        parameters,
        inputs,
        derivatives=True,
    )

    log_normalizer, inclusion = conditional_distribution(
        weights,
        marginals=True,
    )

    log_probability = (
        np.sum(targets * np.log(weights), axis=1)
        - log_normalizer
    )

    nll = -float(np.mean(log_probability))

    parameters = np.asarray(parameters, dtype=float)

    penalty = (
        0.5 * ridge * float(np.mean(parameters ** 2))
    )

    value = nll + penalty

    samples = len(inputs)
    residual = (inclusion - targets) / samples

    output_adjoint = np.zeros_like(output)

    output_adjoint[:, :NUMBERS] = (
        2.0
        * output[:, :NUMBERS]
        * (residual / weights)
    )

    first, second, factor_2 = intermediates

    second_adjoint = (
        output_adjoint @ source.matrix_2.conj()
    )

    gradient_2 = np.real(
        np.sum(
            np.conj(second_adjoint) * (1j * second),
            axis=0,
        )
    )

    middle_adjoint = (
        second_adjoint * factor_2.conj()[None, :]
    )

    first_adjoint = (
        middle_adjoint @ source.matrix_1.conj()
    )

    gradient_1 = np.real(
        np.sum(
            np.conj(first_adjoint) * (1j * first),
            axis=0,
        )
    )

    gradient = np.concatenate(
        (gradient_1, gradient_2)
    )

    gradient += ridge * parameters / PARAMETERS

    require(
        np.isfinite(value),
        "Funkcija cilja nije konačna.",
    )

    check_finite(gradient, "Gradijent")

    return float(value), gradient


def projected_gradient_norm(parameters, gradient):
    projected = np.array(gradient, dtype=float, copy=True)

    lower = parameters <= -math.pi + 1.0e-10
    upper = parameters >= math.pi - 1.0e-10

    projected[lower & (projected > 0.0)] = 0.0
    projected[upper & (projected < 0.0)] = 0.0

    return float(np.max(np.abs(projected)))


def load_draws(path):
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()

    rows = []
    empty_rows = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.reader(handle)

        for line_number, row in enumerate(reader, start=1):
            if not row or all(not field.strip() for field in row):
                empty_rows += 1
                continue

            require(
                len(row) == PICKS,
                f"{path.name}, red {line_number}: očekujem sedam kolona.",
            )

            try:
                values = [int(field.strip()) for field in row]
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(values)) == PICKS,
                f"{path.name}, red {line_number}: ponovljen broj.",
            )

            require(
                all(1 <= value <= NUMBERS for value in values),
                f"{path.name}, red {line_number}: broj izvan 1–39.",
            )

            rows.append(sorted(value - 1 for value in values))

    require(
        len(rows) >= 100,
        f"{path.name}: premalo redova za hronološke podele.",
    )

    draws = np.asarray(rows, dtype=np.int64)

    occupancy = np.zeros(
        (len(draws), NUMBERS),
        dtype=float,
    )

    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1.0

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {empty_rows}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, occupancy, fingerprint


def causal_features(occupancy):
    features = np.zeros_like(occupancy, dtype=float)

    for lag, coefficient in enumerate(HISTORY_COEFFICIENTS):
        if lag == 0:
            features += coefficient * occupancy
        else:
            features[lag:] += (
                coefficient * occupancy[:-lag]
            )

    check_finite(features, "Hronološke karakteristike")

    return features


def checkpoint_path(source, inputs, targets, ridge, initial):
    digest = hashlib.sha256()

    configuration = (
        VERSION,
        str(qiskit.__version__),
        str(np.__version__),
        str(scipy.__version__),
        str(float(ridge)),
        str(MAX_ITERATIONS),
        str(MAX_FUNCTION_CALLS),
        str(HISTORY_COEFFICIENTS),
        str(INPUT_PHASE_SCALE),
    )

    digest.update(repr(configuration).encode("utf-8"))

    for values in (
        inputs,
        targets,
        initial,
        source.matrix_1,
        source.matrix_2,
        BASE_PHASES_1,
        BASE_PHASES_2,
    ):
        array = np.ascontiguousarray(values)

        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())

    return (
        CHECKPOINT_DIRECTORY
        / f"{digest.hexdigest()}.npz"
    )


def save_checkpoint(path, parameters, iterations, done, message):
    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            version=np.asarray(VERSION),
            parameters=np.asarray(parameters, dtype=float),
            iterations=np.asarray(iterations, dtype=np.int64),
            done=np.asarray(bool(done)),
            message=np.asarray(str(message)),
        )

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


def read_checkpoint(path):
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as saved:
            require(
                str(saved["version"].item()) == VERSION,
                "Nepodudarna verzija kontrolne tačke.",
            )

            parameters = np.asarray(
                saved["parameters"],
                dtype=float,
            ).copy()

            iterations = int(saved["iterations"].item())
            done = bool(saved["done"].item())
            message = str(saved["message"].item())

        require(
            parameters.shape == (PARAMETERS,)
            and np.all(np.isfinite(parameters))
            and np.all(np.abs(parameters) <= math.pi + 1.0e-10),
            "Neispravni parametri kontrolne tačke.",
        )

        require(
            0 <= iterations <= MAX_ITERATIONS,
            "Neispravan broj sačuvanih iteracija.",
        )

        return parameters, iterations, done, message

    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        BadZipFile,
        EOFError,
    ) as error:
        say(
            f"Kontrolna tačka nije upotrebljiva: {path.name}",
            str(error),
        )

        return None


@dataclass
class Fitted:
    parameters: np.ndarray
    loss: float
    gradient_norm: float
    converged: bool
    iterations: int
    message: str


def fit_model(
    source,
    inputs,
    targets,
    ridge,
    label,
    initial=None,
):
    require(
        len(inputs) == len(targets) and len(inputs) > 0,
        f"{label}: neispravni trening uzorci.",
    )

    if initial is None:
        initial = source.initial_parameters()

    initial = np.asarray(initial, dtype=float).copy()

    path = checkpoint_path(
        source,
        inputs,
        targets,
        ridge,
        initial,
    )

    saved = read_checkpoint(path)

    parameters = initial.copy()
    used = 0
    done = False
    message = "Početni kandidat"

    if saved is not None:
        parameters, used, done, message = saved

        try:
            objective(
                parameters,
                source,
                inputs,
                targets,
                ridge,
            )
        except (RuntimeError, FloatingPointError, ValueError):
            say(
                f"{label}: sačuvani kandidat nije prošao proveru.",
                "Koristim početni kandidat ovog treninga.",
            )

            parameters = initial.copy()
            used = 0
            done = False
            message = "Ponovljen početak posle neispravne kontrolne tačke"

    value, gradient = objective(
        parameters,
        source,
        inputs,
        targets,
        ridge,
    )

    best = {
        "parameters": parameters.copy(),
        "value": float(value),
        "gradient": gradient.copy(),
    }

    started = time.monotonic()
    prior_iterations = used
    local_iterations = 0

    say(
        "",
        label,
        f"Trening prelaza: {len(inputs)} | regularizacija: {ridge}",
    )

    if saved is not None and done:
        say("Koristim završenu kontrolnu tačku.")
    else:
        save_checkpoint(
            path,
            best["parameters"],
            used,
            False,
            message,
        )

    def fun(candidate):
        candidate_value, candidate_gradient = objective(
            candidate,
            source,
            inputs,
            targets,
            ridge,
        )

        if candidate_value < best["value"]:
            best["parameters"] = np.asarray(
                candidate,
                dtype=float,
            ).copy()

            best["value"] = float(candidate_value)
            best["gradient"] = candidate_gradient.copy()

        return candidate_value, candidate_gradient

    def callback(_candidate):
        nonlocal local_iterations, used

        local_iterations += 1
        used = min(
            MAX_ITERATIONS,
            prior_iterations + local_iterations,
        )

        if local_iterations % CHECKPOINT_EVERY == 0:
            norm = projected_gradient_norm(
                best["parameters"],
                best["gradient"],
            )

            elapsed = time.monotonic() - started

            say(
                f"Iteracija {used}/{MAX_ITERATIONS}"
                f" | proteklo {elapsed:.1f} s"
                f" | cilj {best['value']:.9f}"
                f" | Max|proj.grad| {norm:.3e}"
            )

            save_checkpoint(
                path,
                best["parameters"],
                used,
                False,
                "Trening je u toku",
            )

    remaining = MAX_ITERATIONS - used

    if not done and remaining > 0:
        try:
            result = minimize(
                fun,
                best["parameters"],
                method="L-BFGS-B",
                jac=True,
                bounds=PARAMETER_BOUNDS,
                callback=callback,
                options={
                    "maxiter": remaining,
                    "maxfun": MAX_FUNCTION_CALLS,
                    "maxls": 40,
                    "maxcor": 15,
                    "ftol": 1.0e-12,
                    "gtol": 1.0e-6,
                },
            )

            used = min(
                MAX_ITERATIONS,
                prior_iterations + int(result.nit),
            )

            message = str(result.message)

        except KeyboardInterrupt:
            save_checkpoint(
                path,
                best["parameters"],
                used,
                False,
                "Korisnički prekid",
            )

            raise

        except (
            RuntimeError,
            FloatingPointError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            message = (
                "Optimizacija je zaustavljena; zadržan je najbolji "
                f"validan kandidat. Razlog: {error}"
            )

    elif not done:
        message = "Iskorišćen je budžet iteracija."

    final_value, final_gradient = objective(
        best["parameters"],
        source,
        inputs,
        targets,
        ridge,
    )

    gradient_norm = projected_gradient_norm(
        best["parameters"],
        final_gradient,
    )

    converged = gradient_norm <= GRADIENT_LIMIT

    save_checkpoint(
        path,
        best["parameters"],
        used,
        True,
        message,
    )

    say(
        f"Zadržani cilj: {final_value:.9f}",
        f"Max|proj.grad|: {gradient_norm:.6e}",
        f"Status optimizatora: {message}",
    )

    if not converged:
        say(
            "DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.",
            "Kandidat ima proverenu raspodelu; minimum nije potvrđen.",
        )

    return Fitted(
        parameters=best["parameters"].copy(),
        loss=float(final_value),
        gradient_norm=gradient_norm,
        converged=converged,
        iterations=used,
        message=message,
    )


def evaluate(
    source,
    model,
    inputs,
    targets,
    mode="coherent",
    hits=False,
):
    if mode == "coherent":
        weights, _, _ = source.forward(
            model.parameters,
            inputs,
        )

    elif mode == "zero_actions":
        weights, _, _ = source.forward(
            model.parameters,
            inputs,
            zero_actions=True,
        )

    elif mode == "dephased":
        weights = source.dephased_weights(inputs)

    else:
        raise RuntimeError("Nepoznat način evaluacije.")

    log_probabilities = subset_log_probabilities(
        weights,
        targets,
    )

    nll = -float(np.mean(log_probabilities))

    histogram = None

    if hits:
        selected = np.argsort(
            -weights,
            axis=1,
            kind="stable",
        )[:, :PICKS]

        matches = np.sum(
            np.take_along_axis(
                targets,
                selected,
                axis=1,
            ),
            axis=1,
        ).astype(np.int64)

        histogram = np.bincount(
            matches,
            minlength=PICKS + 1,
        )

        require(
            int(np.sum(histogram)) == len(targets),
            "Neispravna statistika pogodaka.",
        )

    return nll, histogram


def verify_input_preparation(source, features, prepared):
    indices = sorted(
        set((0, len(features) // 2, len(features) - 1))
    )

    zero = np.zeros(PARAMETERS, dtype=float)

    for index in indices:
        phases = source.input_phases(
            features[index:index + 1]
        )[0]

        circuit = source.preparation.copy()

        circuit.unitary(
            np.diag(np.exp(1j * phases)),
            list(range(QUBITS)),
            label="History",
        )

        direct = np.asarray(
            Statevector.from_instruction(circuit).data
        )

        require(
            np.max(
                np.abs(direct - prepared[index])
            ) < 2.0e-10,
            "Paketna priprema ulaza odstupa od Qiskit simulacije.",
        )

        local = source.local_circuit(
            zero,
            features[index],
        )

        direct_output = np.asarray(
            Statevector.from_instruction(local).data
        )

        _, batch_output, _ = source.forward(
            zero,
            prepared[index:index + 1],
        )

        require(
            np.max(
                np.abs(direct_output - batch_output[0])
            ) < 2.0e-10,
            "Paketna evolucija odstupa od Qiskit Statevector simulacije.",
        )


def self_checks(source):
    say(
        "",
        "Pokrećem ugrađene matematičke i Qiskit provere.",
    )

    toy_weights = np.asarray(
        [[0.07, 0.13, 0.19, 0.23, 0.17, 0.21]],
        dtype=float,
    )

    toy_picks = 3

    log_z, marginals = conditional_distribution(
        toy_weights,
        picks=toy_picks,
        marginals=True,
    )

    legal = list(
        combinations(range(toy_weights.shape[1]), toy_picks)
    )

    raw = np.asarray([
        np.prod(toy_weights[0, list(subset)])
        for subset in legal
    ])

    enumerated_z = float(np.sum(raw))
    probabilities = raw / enumerated_z

    require(
        abs(float(log_z[0]) - math.log(enumerated_z)) < 1.0e-11,
        "Normalizacija odstupa od potpune male enumeracije.",
    )

    require(
        abs(float(np.sum(probabilities)) - 1.0) < 1.0e-12,
        "Mala enumeracija nije normalizovana.",
    )

    expected_marginals = np.zeros(toy_weights.shape[1])

    for subset, probability in zip(legal, probabilities):
        expected_marginals[list(subset)] += probability

    require(
        np.max(
            np.abs(
                expected_marginals - marginals[0]
            )
        ) < 1.0e-11,
        "Marginale odstupaju od male enumeracije.",
    )

    selected, log_probability = exact_maximum(
        toy_weights[0],
        picks=toy_picks,
    )

    expected_index = int(np.argmax(raw))

    require(
        selected == legal[expected_index],
        "Izbor maksimuma odstupa od male enumeracije.",
    )

    require(
        abs(
            log_probability
            - math.log(float(probabilities[expected_index]))
        ) < 1.0e-11,
        "Verovatnoća maksimuma je neispravna.",
    )

    for picks in (0, 1, toy_weights.shape[1]):
        _, marginal = conditional_distribution(
            toy_weights,
            picks=picks,
            marginals=True,
        )

        require(
            abs(float(np.sum(marginal)) - picks) < 1.0e-10,
            "Neispravan granični slučaj raspodele.",
        )

    tied, _ = exact_maximum(
        np.ones(6, dtype=float),
        picks=3,
    )

    require(
        tied == (0, 1, 2),
        "Nedeterministički izbor pri jednakim težinama.",
    )

    toy_occupancy = np.zeros((5, NUMBERS), dtype=float)

    for row in range(len(toy_occupancy)):
        numbers = (
            np.arange(PICKS) * 5 + row * 3
        ) % NUMBERS

        toy_occupancy[row, numbers] = 1.0

    features = causal_features(toy_occupancy)
    prepared = source.prepare_inputs(features)

    verify_input_preparation(
        source,
        features,
        prepared,
    )

    train_inputs = prepared[:-1]
    targets = toy_occupancy[1:]

    parameters = (
        0.04 * np.sin(
            np.arange(PARAMETERS, dtype=float) + 0.3
        )
    )

    ridge = 0.1

    _, analytic = objective(
        parameters,
        source,
        train_inputs,
        targets,
        ridge,
    )

    epsilon = 1.0e-6

    directions = []

    for index in (
        0,
        7,
        31,
        63,
        64,
        79,
        103,
        127,
    ):
        direction = np.zeros(PARAMETERS)
        direction[index] = 1.0
        directions.append(direction)

    dense_direction = np.cos(
        np.arange(PARAMETERS, dtype=float) * 0.43 + 0.2
    )

    dense_direction /= np.linalg.norm(dense_direction)
    directions.append(dense_direction)

    maximum_gradient_error = 0.0

    for direction in directions:
        plus, _ = objective(
            parameters + epsilon * direction,
            source,
            train_inputs,
            targets,
            ridge,
        )

        minus, _ = objective(
            parameters - epsilon * direction,
            source,
            train_inputs,
            targets,
            ridge,
        )

        numerical = (plus - minus) / (2.0 * epsilon)
        expected = float(analytic @ direction)

        error = abs(numerical - expected)
        maximum_gradient_error = max(
            maximum_gradient_error,
            error,
        )

        require(
            error < 3.0e-5 * (1.0 + abs(expected)),
            "Analitički gradijent ne prolazi numeričku proveru.",
        )

    prefix = causal_features(toy_occupancy[:3])

    require(
        np.array_equal(prefix, features[:3]),
        "Karakteristike koriste buduće podatke.",
    )

    seven = source.seven_register_circuit(
        parameters,
        features[-1],
    )

    require(
        seven.num_qubits == 42,
        "Konstrukcija sedam registara je neispravna.",
    )

    say(
        "Normalizacija i margine: potvrđene na maloj enumeraciji.",
        "Maksimum i granični slučajevi: potvrđeni.",
        "Paketna evolucija i Qiskit Statevector: usaglašeni.",
        f"Najveća proverena greška gradijenta: {maximum_gradient_error:.3e}",
        "Hronološka konstrukcija ulaza: potvrđena.",
        "Sedam registara: 42 kubita u kompaktnoj proizvodnoj reprezentaciji.",
    )


def verify_next(source, model, features_row, prepared_row, selected):
    weights, output, _ = source.forward(
        model.parameters,
        prepared_row[None, :],
    )

    local = source.local_circuit(
        model.parameters,
        features_row,
    )

    direct_error = 0.0

    # Sedam lokalnih izvršavanja odgovara proizvodu sedam registara.
    # Ne pravi se gust vektor od 2**42 amplituda.
    for _register_index in range(PICKS):
        direct = np.asarray(
            Statevector.from_instruction(local).data
        )

        direct_error = max(
            direct_error,
            float(np.max(np.abs(direct - output[0]))),
        )

    require(
        direct_error < 2.0e-10,
        "NEXT amplitude odstupaju od Qiskit Statevector izvršavanja.",
    )

    factor_1, factor_2 = source.phase_factors(
        model.parameters
    )

    path_error = 0.0
    selected_coherent = []
    selected_incoherent = []

    for number in selected:
        # terms[z, x] je doprinos putanje:
        # početno x -> međustanje z -> izlazni broj.
        terms = (
            source.matrix_2[number, :, None]
            * factor_2[:, None]
            * source.matrix_1
            * factor_1[None, :]
            * prepared_row[None, :]
        )

        summed_amplitude = np.sum(terms)

        path_error = max(
            path_error,
            float(
                abs(
                    summed_amplitude - output[0, number]
                )
            ),
        )

        selected_coherent.append(
            float(abs(summed_amplitude) ** 2)
        )

        selected_incoherent.append(
            float(np.sum(np.abs(terms) ** 2))
        )

    require(
        path_error < 2.0e-10,
        "Eksplicitni zbir putanja odstupa od Qiskit amplitude.",
    )

    dephased = source.dephased_weights(
        prepared_row[None, :]
    )[0]

    require(
        np.max(
            np.abs(
                dephased[list(selected)]
                - np.asarray(selected_incoherent)
            )
        ) < 2.0e-10,
        "Nekoherentni zbir putanja odstupa od dekoherentne kontrole.",
    )

    log_z, _ = conditional_distribution(
        weights,
        marginals=False,
    )

    alternative_log_z = independent_normalizer_check(
        weights[0]
    )

    normalization_error = abs(
        alternative_log_z - float(log_z[0])
    )

    require(
        normalization_error < 2.0e-10,
        "Dve numeričke normalizacije NEXT raspodele se ne slažu.",
    )

    normalization_ratio = math.exp(
        alternative_log_z - float(log_z[0])
    )

    interference_difference = float(
        np.max(
            np.abs(
                np.asarray(selected_coherent)
                - np.asarray(selected_incoherent)
            )
        )
    )

    return {
        "qiskit_error": direct_error,
        "path_error": path_error,
        "normalization_error": normalization_error,
        "normalization_ratio": normalization_ratio,
        "interference_difference": interference_difference,
    }


def longest_run(numbers):
    best = 0
    current = 0
    previous = None

    for number in sorted(numbers):
        if previous is not None and number == previous + 1:
            current += 1
        else:
            current = 1

        best = max(best, current)
        previous = number

    return best


def print_result(result):
    say(
        "",
        result["name"],
        "NEXT: " + " ".join(
            str(number + 1)
            for number in result["selected"]
        ),
        f"Validacioni NLL: {result['validation_nll']:.9f}",
        f"Odvojeni test NLL: {result['test_nll']:.9f}",
        (
            "Test NLL bez faza dve akcione kapije: "
            f"{result['zero_actions_nll']:.9f}"
        ),
        (
            "Test NLL sa uklonjenom koherencijom putanja: "
            f"{result['dephased_nll']:.9f}"
        ),
        f"Modelna verovatnoća NEXT: {result['next_probability']:.12g}",
        f"Normalizacioni odnos: {result['normalization_ratio']:.12f}",
        f"Greška dve normalizacije: {result['normalization_error']:.3e}",
        f"Greška Qiskit amplituda: {result['qiskit_error']:.3e}",
        f"Greška eksplicitnog zbira putanja: {result['path_error']:.3e}",
        (
            "Razlika koherentnog i nekoherentnog zbira za NEXT brojeve: "
            f"{result['interference_difference']:.6e}"
        ),
        f"Dijagnostička osetljivost na istoriju: {result['sensitivity']:.6e}",
        f"Zbir marginalnih verovatnoća uključivanja: {result['marginal_sum']:.12f}",
        f"Završni Max|proj.grad|: {result['gradient_norm']:.6e}",
        "Test: broj kola prema broju pogođenih brojeva",
        " | ".join(
            f"{hits}: {int(count)}"
            for hits, count in enumerate(result["histogram"])
        ),
        f"Prosečan broj pogodaka na testu: {result['mean_hits']:.6f}",
        "Marginalna verovatnoća uključivanja svakog broja:",
    )

    inclusion = result["inclusion"]

    for start in range(0, NUMBERS, 6):
        stop = min(start + 6, NUMBERS)

        say(
            " | ".join(
                f"{number + 1:02d}: {inclusion[number]:.8f}"
                for number in range(start, stop)
            )
        )

    for warning in result["warnings"]:
        say(f"DIJAGNOSTIKA: {warning}")


def process_file(source, path, draws, occupancy, fingerprint):
    del fingerprint

    features = causal_features(occupancy)
    prepared = source.prepare_inputs(features)

    verify_input_preparation(
        source,
        features,
        prepared,
    )

    inputs = prepared[:-1]
    targets = occupancy[1:]

    count = len(inputs)

    split_60 = int(0.60 * count)
    split_70 = int(0.70 * count)
    split_80 = int(0.80 * count)

    require(
        20 < split_60 < split_70 < split_80 < count,
        f"{path.name}: neispravne hronološke podele.",
    )

    folds = (
        (split_60, split_70),
        (split_70, split_80),
    )

    validation_records = []
    unconfirmed_fits = 0

    for ridge in RIDGES:
        weighted_sum = 0.0
        validation_count = 0

        warm = source.initial_parameters()

        for fold_index, (train_end, validation_end) in enumerate(
            folds,
            start=1,
        ):
            model = fit_model(
                source,
                inputs[:train_end],
                targets[:train_end],
                ridge,
                (
                    f"{path.name} / validacija {fold_index}"
                    f" / ridge {ridge}"
                ),
                initial=warm,
            )

            if not model.converged:
                unconfirmed_fits += 1

            validation_nll, _ = evaluate(
                source,
                model,
                inputs[train_end:validation_end],
                targets[train_end:validation_end],
            )

            fold_count = validation_end - train_end

            weighted_sum += validation_nll * fold_count
            validation_count += fold_count
            warm = model.parameters.copy()

            say(
                f"Validacija {fold_index} NLL: {validation_nll:.9f}"
            )

        average_nll = weighted_sum / validation_count

        validation_records.append(
            (
                float(average_nll),
                float(ridge),
                warm.copy(),
            )
        )

        say(
            f"Objedinjeni validacioni NLL za ridge {ridge}: "
            f"{average_nll:.9f}"
        )

    validation_records.sort(
        key=lambda record: (record[0], record[1])
    )

    validation_nll, chosen_ridge, warm = validation_records[0]

    say(
        "",
        f"{path.name}: izabrana regularizacija {chosen_ridge}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    test_model = fit_model(
        source,
        inputs[:split_80],
        targets[:split_80],
        chosen_ridge,
        f"{path.name} / model za odvojeni test",
        initial=warm,
    )

    if not test_model.converged:
        unconfirmed_fits += 1

    test_inputs = inputs[split_80:]
    test_targets = targets[split_80:]

    test_nll, histogram = evaluate(
        source,
        test_model,
        test_inputs,
        test_targets,
        mode="coherent",
        hits=True,
    )

    zero_actions_nll, _ = evaluate(
        source,
        test_model,
        test_inputs,
        test_targets,
        mode="zero_actions",
    )

    dephased_nll, _ = evaluate(
        source,
        test_model,
        test_inputs,
        test_targets,
        mode="dephased",
    )

    say(
        "",
        f"{path.name}: odvojeni test je obrađen.",
        f"Test NLL: {test_nll:.9f}",
        "Kontrole menjaju isto kvantno kolo i nisu posebno trenirani konkurenti.",
        "Test kontrole ne učestvuju u izboru regularizacije.",
    )

    final_model = fit_model(
        source,
        inputs,
        targets,
        chosen_ridge,
        f"{path.name} / završno učenje na svim prelazima",
        initial=test_model.parameters,
    )

    if not final_model.converged:
        unconfirmed_fits += 1

    next_weights, _, _ = source.forward(
        final_model.parameters,
        prepared[-1:],
    )

    _, inclusion = conditional_distribution(
        next_weights,
        marginals=True,
    )

    selected, next_log_probability = exact_maximum(
        next_weights[0]
    )

    require(
        len(selected) == PICKS
        and len(set(selected)) == PICKS
        and all(0 <= number < NUMBERS for number in selected),
        "NEXT nije legalna kombinacija.",
    )

    verification = verify_next(
        source,
        final_model,
        features[-1],
        prepared[-1],
        selected,
    )

    diagnostic_inputs = prepared[-min(8, len(prepared)):]

    diagnostic_weights, _, _ = source.forward(
        final_model.parameters,
        diagnostic_inputs,
    )

    _, diagnostic_inclusion = conditional_distribution(
        diagnostic_weights,
        marginals=True,
    )

    sensitivity = float(
        np.max(
            np.ptp(
                diagnostic_inclusion,
                axis=0,
            )
        )
    )

    warnings = []

    if unconfirmed_fits:
        warnings.append(
            f"{unconfirmed_fits} treninga nema potvrđen gradijentni uslov; "
            "njihovi rezultati ne potvrđuju dostizanje minimuma."
        )

    if test_nll >= zero_actions_nll:
        warnings.append(
            "Učene akcione faze nisu dale niži test NLL "
            "od dijagnostičke kontrole sa nuliranim akcionim fazama."
        )

    if test_nll >= dephased_nll:
        warnings.append(
            "Koherentni model nije dao niži test NLL "
            "od dijagnostičke kontrole bez koherencije."
        )

    run = longest_run(selected)

    if run >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run} uzastopnih brojeva."
        )

    if selected[-1] - selected[0] <= 9:
        warnings.append(
            "Svih sedam NEXT brojeva nalazi se u rasponu od deset brojeva."
        )

    if np.any(
        np.all(
            draws == np.asarray(selected)[None, :],
            axis=1,
        )
    ):
        warnings.append(
            "NEXT se već pojavio u istoriji ovog CSV-a."
        )

    if np.array_equal(
        np.asarray(selected),
        draws[-1],
    ):
        warnings.append(
            "NEXT je jednak poslednjem kolu."
        )

    if sensitivity < 1.0e-8:
        warnings.append(
            "Raspodela slabo reaguje na poslednje proverene istorijske ulaze."
        )

    if verification["interference_difference"] < 1.0e-10:
        warnings.append(
            "Za izabrane NEXT brojeve izmerena interferentna razlika je mala."
        )

    mean_hits = float(
        np.dot(
            np.arange(PICKS + 1, dtype=float),
            histogram,
        ) / np.sum(histogram)
    )

    result = {
        "name": path.name,
        "selected": selected,
        "validation_nll": validation_nll,
        "test_nll": test_nll,
        "zero_actions_nll": zero_actions_nll,
        "dephased_nll": dephased_nll,
        "next_probability": math.exp(next_log_probability),
        "inclusion": inclusion[0].copy(),
        "marginal_sum": float(np.sum(inclusion[0])),
        "sensitivity": sensitivity,
        "gradient_norm": final_model.gradient_norm,
        "histogram": histogram,
        "mean_hits": mean_hits,
        "warnings": warnings,
        **verification,
    }

    print_result(result)

    return result


def main():
    say(
        "QUANTUM PUTANJE V1 — Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "",
        "Osnova: zbir kompleksnih amplituda putanja i interferencija.",
        "Uče se fazne korekcije dve dijagonalne kvantne kapije.",
        "Izlaz nastaje iz operatora stvarnog Qiskit kola.",
        "Operator se paketno primenjuje na istorijska kvantna stanja.",
        "Završni izlaz se dodatno proverava Qiskit Statevector simulacijom.",
        "",
        "Sedam registara koristi po šest kubita i isto naučeno kolo.",
        "Registri imaju proizvodno stanje; nema međuregistarskog sprezanja.",
        "Uslov sedam različitih brojeva obrađuje se tačnom normalizacijom.",
        "Zajednička raspodela nema dodatno naučene interakcije parova brojeva.",
        "Izbor maksimuma ove raspodele je tačan.",
        "",
        "Optimizator, gradijent i normalizacija su klasični proračuni.",
        "Nema klasičnog regresora koji zamenjuje kvantne amplitude.",
        "Nema slučajnog generisanja niti uzorkovanja merenja.",
        "Nema filtera parnosti, zbira, zona ili uzastopnih brojeva.",
        "Optimizacija je lokalna; globalni minimum nije potvrđen.",
        "",
        "Kontrole uklanjaju akcione faze ili koherenciju istog kola.",
        "Kontrole su dijagnostičke i nisu posebno trenirani konkurenti.",
        "Niži NLL znači bolju dodeljenu verovatnoću stvarnim sedmorkama.",
        "",
        f"Kontrolne tačke: {CHECKPOINT_DIRECTORY}",
    )

    source = QuantumPaths()

    loaded = []

    for path in FILES:
        draws, occupancy, fingerprint = load_draws(path)

        loaded.append(
            (path, draws, occupancy, fingerprint)
        )

    self_checks(source)

    CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for path, draws, occupancy, fingerprint in loaded:
        result = process_file(
            source,
            path,
            draws,
            occupancy,
            fingerprint,
        )

        results.append(result)

    require(
        len(results) == len(FILES),
        "Nisu završena sva tri ulaza.",
    )

    say(
        "",
        "ZAVRŠNI NEXT REZULTATI",
    )

    for result in results:
        say(
            result["name"] + " | NEXT: "
            + " ".join(
                str(number + 1)
                for number in result["selected"]
            )
        )

    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            if (
                results[first]["selected"]
                == results[second]["selected"]
            ):
                say(
                    "DIJAGNOSTIKA: isti NEXT za "
                    f"{results[first]['name']} i "
                    f"{results[second]['name']}.",
                    "Rezultat nije menjan radi veštačkog razlikovanja ulaza.",
                    (
                        "Osetljivosti na istoriju: "
                        f"{results[first]['sensitivity']:.6e} i "
                        f"{results[second]['sensitivity']:.6e}."
                    ),
                )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        say(
            "",
            "PREKINUTO na zahtev korisnika.",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say(
            "",
            f"ZAUSTAVLJENO: {error}",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM PUTANJE V1 — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4

Osnova: zbir kompleksnih amplituda putanja i interferencija.
Uče se fazne korekcije dve dijagonalne kvantne kapije.
Izlaz nastaje iz operatora stvarnog Qiskit kola.
Operator se paketno primenjuje na istorijska kvantna stanja.
Završni izlaz se dodatno proverava Qiskit Statevector simulacijom.

Sedam registara koristi po šest kubita i isto naučeno kolo.
Registri imaju proizvodno stanje; nema međuregistarskog sprezanja.
Uslov sedam različitih brojeva obrađuje se tačnom normalizacijom.
Zajednička raspodela nema dodatno naučene interakcije parova brojeva.
Izbor maksimuma ove raspodele je tačan.

Optimizator, gradijent i normalizacija su klasični proračuni.
Nema klasičnog regresora koji zamenjuje kvantne amplitude.
Nema slučajnog generisanja niti uzorkovanja merenja.
Nema filtera parnosti, zbira, zona ili uzastopnih brojeva.
Optimizacija je lokalna; globalni minimum nije potvrđen.

Kontrole uklanjaju akcione faze ili koherenciju istog kola.
Kontrole su dijagnostičke i nisu posebno trenirani konkurenti.
Niži NLL znači bolju dodeljenu verovatnoću stvarnim sedmorkama.

Kontrolne tačke: /Quantum_Putanje/Quantum_Putanje_V1_checkpoints

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled iz CSV-a.

Pokrećem ugrađene matematičke i Qiskit provere.
Normalizacija i margine: potvrđene na maloj enumeraciji.
Maksimum i granični slučajevi: potvrđeni.
Paketna evolucija i Qiskit Statevector: usaglašeni.
Najveća proverena greška gradijenta: 2.867e-09
Hronološka konstrukcija ulaza: potvrđena.
Sedam registara: 42 kubita u kompaktnoj proizvodnoj reprezentaciji.

loto7_4710_k86.csv / validacija 1 / ridge 0.03
Trening prelaza: 2825 | regularizacija: 0.03
Iteracija 10/160 | proteklo 0.8 s | cilj 17.171950945 | Max|proj.grad| 5.905e-02
Iteracija 20/160 | proteklo 1.4 s | cilj 16.937997429 | Max|proj.grad| 4.943e-02
Iteracija 30/160 | proteklo 2.0 s | cilj 16.855595614 | Max|proj.grad| 1.617e-02
Iteracija 40/160 | proteklo 2.6 s | cilj 16.803676833 | Max|proj.grad| 8.866e-03
Iteracija 50/160 | proteklo 3.3 s | cilj 16.769852297 | Max|proj.grad| 2.864e-02
Iteracija 60/160 | proteklo 3.9 s | cilj 16.745880586 | Max|proj.grad| 1.021e-02
Iteracija 70/160 | proteklo 4.5 s | cilj 16.727598974 | Max|proj.grad| 1.258e-02
Iteracija 80/160 | proteklo 5.1 s | cilj 16.718092463 | Max|proj.grad| 2.790e-02
Iteracija 90/160 | proteklo 5.8 s | cilj 16.712965719 | Max|proj.grad| 2.310e-02
Iteracija 100/160 | proteklo 6.4 s | cilj 16.709935623 | Max|proj.grad| 2.480e-03
Iteracija 110/160 | proteklo 7.0 s | cilj 16.707917320 | Max|proj.grad| 4.619e-03
Iteracija 120/160 | proteklo 7.6 s | cilj 16.706836275 | Max|proj.grad| 9.801e-03
Iteracija 130/160 | proteklo 8.2 s | cilj 16.705822556 | Max|proj.grad| 1.422e-03
Iteracija 140/160 | proteklo 8.8 s | cilj 16.705318294 | Max|proj.grad| 8.946e-04
Iteracija 150/160 | proteklo 9.4 s | cilj 16.704923251 | Max|proj.grad| 6.938e-03
Iteracija 160/160 | proteklo 10.0 s | cilj 16.704598529 | Max|proj.grad| 8.509e-04
Zadržani cilj: 16.704598529
Max|proj.grad|: 8.508603e-04
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 16.781956149

loto7_4710_k86.csv / validacija 2 / ridge 0.03
Trening prelaza: 3296 | regularizacija: 0.03
Iteracija 10/160 | proteklo 0.8 s | cilj 16.715902726 | Max|proj.grad| 2.665e-03
Iteracija 20/160 | proteklo 1.6 s | cilj 16.715011344 | Max|proj.grad| 2.426e-03
Iteracija 30/160 | proteklo 2.3 s | cilj 16.714521661 | Max|proj.grad| 9.830e-04
Iteracija 40/160 | proteklo 3.0 s | cilj 16.714232147 | Max|proj.grad| 1.333e-03
Iteracija 50/160 | proteklo 3.7 s | cilj 16.713735111 | Max|proj.grad| 2.800e-03
Iteracija 60/160 | proteklo 4.4 s | cilj 16.713190360 | Max|proj.grad| 9.649e-04
Iteracija 70/160 | proteklo 5.1 s | cilj 16.712913093 | Max|proj.grad| 1.741e-03
Iteracija 80/160 | proteklo 5.9 s | cilj 16.712814836 | Max|proj.grad| 2.870e-04
Iteracija 90/160 | proteklo 6.6 s | cilj 16.712760343 | Max|proj.grad| 1.026e-03
Iteracija 100/160 | proteklo 7.3 s | cilj 16.712732272 | Max|proj.grad| 1.077e-03
Iteracija 110/160 | proteklo 8.0 s | cilj 16.712714522 | Max|proj.grad| 3.026e-04
Iteracija 120/160 | proteklo 8.7 s | cilj 16.712705809 | Max|proj.grad| 1.498e-04
Iteracija 130/160 | proteklo 9.5 s | cilj 16.712702052 | Max|proj.grad| 5.522e-05
Iteracija 140/160 | proteklo 10.3 s | cilj 16.712700452 | Max|proj.grad| 1.058e-04
Iteracija 150/160 | proteklo 11.0 s | cilj 16.712699132 | Max|proj.grad| 9.650e-05
Iteracija 160/160 | proteklo 11.7 s | cilj 16.712698634 | Max|proj.grad| 2.122e-05
Zadržani cilj: 16.712698634
Max|proj.grad|: 2.121631e-05
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Validacija 2 NLL: 16.783638391
Objedinjeni validacioni NLL za ridge 0.03: 16.782797270

loto7_4710_k86.csv / validacija 1 / ridge 0.3
Trening prelaza: 2825 | regularizacija: 0.3
Iteracija 10/160 | proteklo 0.7 s | cilj 17.159084562 | Max|proj.grad| 4.818e-02
Iteracija 20/160 | proteklo 1.2 s | cilj 16.991975412 | Max|proj.grad| 2.462e-02
Iteracija 30/160 | proteklo 1.8 s | cilj 16.910857606 | Max|proj.grad| 1.185e-02
Iteracija 40/160 | proteklo 2.4 s | cilj 16.886736876 | Max|proj.grad| 1.804e-02
Iteracija 50/160 | proteklo 3.1 s | cilj 16.867521535 | Max|proj.grad| 1.818e-02
Iteracija 60/160 | proteklo 3.7 s | cilj 16.863897331 | Max|proj.grad| 1.966e-03
Iteracija 70/160 | proteklo 4.4 s | cilj 16.861932851 | Max|proj.grad| 4.015e-03
Iteracija 80/160 | proteklo 5.0 s | cilj 16.860718789 | Max|proj.grad| 1.811e-03
Iteracija 90/160 | proteklo 5.6 s | cilj 16.859966127 | Max|proj.grad| 2.565e-03
Iteracija 100/160 | proteklo 6.3 s | cilj 16.859384781 | Max|proj.grad| 1.354e-03
Iteracija 110/160 | proteklo 7.0 s | cilj 16.858773438 | Max|proj.grad| 1.903e-03
Iteracija 120/160 | proteklo 7.6 s | cilj 16.858421595 | Max|proj.grad| 2.552e-03
Iteracija 130/160 | proteklo 8.2 s | cilj 16.858003506 | Max|proj.grad| 9.733e-04
Iteracija 140/160 | proteklo 8.9 s | cilj 16.857433316 | Max|proj.grad| 2.352e-03
Iteracija 150/160 | proteklo 9.5 s | cilj 16.856935994 | Max|proj.grad| 1.276e-03
Iteracija 160/160 | proteklo 10.0 s | cilj 16.856482881 | Max|proj.grad| 2.847e-03
Zadržani cilj: 16.856482881
Max|proj.grad|: 2.847349e-03
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 16.841831627

loto7_4710_k86.csv / validacija 2 / ridge 0.3
Trening prelaza: 3296 | regularizacija: 0.3
Iteracija 10/160 | proteklo 0.8 s | cilj 16.868742931 | Max|proj.grad| 2.412e-03
Iteracija 20/160 | proteklo 1.5 s | cilj 16.867993913 | Max|proj.grad| 5.077e-03
Iteracija 30/160 | proteklo 2.3 s | cilj 16.867735281 | Max|proj.grad| 3.338e-03
Iteracija 40/160 | proteklo 3.0 s | cilj 16.867343537 | Max|proj.grad| 9.940e-04
Iteracija 50/160 | proteklo 3.7 s | cilj 16.866070444 | Max|proj.grad| 1.498e-03
Iteracija 60/160 | proteklo 4.5 s | cilj 16.864462478 | Max|proj.grad| 4.089e-03
Iteracija 70/160 | proteklo 5.2 s | cilj 16.862861955 | Max|proj.grad| 4.012e-03
Iteracija 80/160 | proteklo 6.1 s | cilj 16.862218689 | Max|proj.grad| 3.525e-03
Iteracija 90/160 | proteklo 6.9 s | cilj 16.861928443 | Max|proj.grad| 5.727e-04
Iteracija 100/160 | proteklo 7.7 s | cilj 16.861743494 | Max|proj.grad| 1.432e-03
Iteracija 110/160 | proteklo 8.4 s | cilj 16.861521500 | Max|proj.grad| 6.481e-03
Iteracija 120/160 | proteklo 9.2 s | cilj 16.861041532 | Max|proj.grad| 1.627e-03
Iteracija 130/160 | proteklo 10.0 s | cilj 16.860719693 | Max|proj.grad| 1.492e-03
Iteracija 140/160 | proteklo 10.7 s | cilj 16.860606377 | Max|proj.grad| 4.079e-04
Iteracija 150/160 | proteklo 11.4 s | cilj 16.860556475 | Max|proj.grad| 7.581e-04
Iteracija 160/160 | proteklo 12.1 s | cilj 16.860517788 | Max|proj.grad| 2.742e-04
Zadržani cilj: 16.860517788
Max|proj.grad|: 2.741865e-04
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.
Validacija 2 NLL: 16.815086891
Objedinjeni validacioni NLL za ridge 0.3: 16.828459259

loto7_4710_k86.csv: izabrana regularizacija 0.03
Validacioni NLL: 16.782797270

loto7_4710_k86.csv / model za odvojeni test
Trening prelaza: 3767 | regularizacija: 0.03
Iteracija 10/160 | proteklo 1.0 s | cilj 16.721711549 | Max|proj.grad| 1.409e-03
Iteracija 20/160 | proteklo 1.8 s | cilj 16.720750822 | Max|proj.grad| 2.288e-03
Iteracija 30/160 | proteklo 2.6 s | cilj 16.720280899 | Max|proj.grad| 3.426e-03
Iteracija 40/160 | proteklo 3.5 s | cilj 16.719820508 | Max|proj.grad| 6.251e-03
Iteracija 50/160 | proteklo 4.3 s | cilj 16.719397477 | Max|proj.grad| 2.139e-03
Iteracija 60/160 | proteklo 5.0 s | cilj 16.718928443 | Max|proj.grad| 1.674e-03
Iteracija 70/160 | proteklo 5.9 s | cilj 16.718666906 | Max|proj.grad| 5.507e-04
Iteracija 80/160 | proteklo 6.7 s | cilj 16.718563734 | Max|proj.grad| 4.581e-04
Iteracija 90/160 | proteklo 7.5 s | cilj 16.718521766 | Max|proj.grad| 1.667e-04
Iteracija 100/160 | proteklo 8.3 s | cilj 16.718512372 | Max|proj.grad| 1.435e-04
Iteracija 110/160 | proteklo 9.1 s | cilj 16.718506996 | Max|proj.grad| 1.294e-03
Iteracija 120/160 | proteklo 9.9 s | cilj 16.718502882 | Max|proj.grad| 1.701e-04
Iteracija 130/160 | proteklo 10.8 s | cilj 16.718498721 | Max|proj.grad| 1.402e-04
Iteracija 140/160 | proteklo 11.7 s | cilj 16.718495526 | Max|proj.grad| 1.772e-04
Iteracija 150/160 | proteklo 12.4 s | cilj 16.718493046 | Max|proj.grad| 8.542e-05
Iteracija 160/160 | proteklo 13.2 s | cilj 16.718490373 | Max|proj.grad| 1.235e-04
Zadržani cilj: 16.718490373
Max|proj.grad|: 1.235131e-04
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.

loto7_4710_k86.csv: odvojeni test je obrađen.
Test NLL: 16.780255853
Kontrole menjaju isto kvantno kolo i nisu posebno trenirani konkurenti.
Test kontrole ne učestvuju u izboru regularizacije.

loto7_4710_k86.csv / završno učenje na svim prelazima
Trening prelaza: 4709 | regularizacija: 0.03
Iteracija 10/160 | proteklo 1.1 s | cilj 16.730735967 | Max|proj.grad| 2.026e-03
Iteracija 20/160 | proteklo 2.0 s | cilj 16.729513008 | Max|proj.grad| 2.377e-03
Iteracija 30/160 | proteklo 3.1 s | cilj 16.729052818 | Max|proj.grad| 2.086e-03
Iteracija 40/160 | proteklo 4.1 s | cilj 16.728754638 | Max|proj.grad| 8.629e-04
Iteracija 50/160 | proteklo 5.1 s | cilj 16.728487650 | Max|proj.grad| 1.665e-03
Iteracija 60/160 | proteklo 6.1 s | cilj 16.727739370 | Max|proj.grad| 2.410e-03
Iteracija 70/160 | proteklo 7.1 s | cilj 16.726958743 | Max|proj.grad| 2.571e-03
Iteracija 80/160 | proteklo 8.2 s | cilj 16.726721802 | Max|proj.grad| 1.809e-03
Iteracija 90/160 | proteklo 9.2 s | cilj 16.726526177 | Max|proj.grad| 5.401e-03
Iteracija 100/160 | proteklo 10.3 s | cilj 16.726406646 | Max|proj.grad| 6.100e-04
Iteracija 110/160 | proteklo 11.4 s | cilj 16.726302151 | Max|proj.grad| 4.065e-04
Iteracija 120/160 | proteklo 12.4 s | cilj 16.726114140 | Max|proj.grad| 8.286e-04
Iteracija 130/160 | proteklo 13.4 s | cilj 16.725843757 | Max|proj.grad| 1.895e-03
Iteracija 140/160 | proteklo 14.5 s | cilj 16.725558353 | Max|proj.grad| 9.776e-04
Iteracija 150/160 | proteklo 15.4 s | cilj 16.725363198 | Max|proj.grad| 6.698e-04
Iteracija 160/160 | proteklo 16.4 s | cilj 16.725215366 | Max|proj.grad| 7.012e-04
Zadržani cilj: 16.725215366
Max|proj.grad|: 7.011691e-04
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.

loto7_4710_k86.csv
NEXT: 5 x 18 y 22 z 35
Validacioni NLL: 16.782797270
Odvojeni test NLL: 16.780255853
Test NLL bez faza dve akcione kapije: 18.835062675
Test NLL sa uklonjenom koherencijom putanja: 16.617673609
Modelna verovatnoća NEXT: 5.79508724433e-07
Normalizacioni odnos: 1.000000000000
Greška dve normalizacije: 3.553e-15
Greška Qiskit amplituda: 1.367e-16
Greška eksplicitnog zbira putanja: 1.279e-16
Razlika koherentnog i nekoherentnog zbira za NEXT brojeve: 1.978645e-02
Dijagnostička osetljivost na istoriju: 1.729303e-01
Zbir marginalnih verovatnoća uključivanja: 7.000000000000
Završni Max|proj.grad|: 7.011691e-04
Test: broj kola prema broju pogođenih brojeva
0: 213 | 1: 377 | 2: 261 | 3: 81 | 4: 8 | 5: 2 | 6: 0 | 7: 0
Prosečan broj pogodaka na testu: 1.256900
Marginalna verovatnoća uključivanja svakog broja:
01: 0.16671575 | 02: 0.12878832 | 03: 0.15656995 | 04: 0.17663819 | 05: 0.23578374 | 06: 0.16860844
07: 0.15970371 | 08: 0.20054959 | 09: 0.15620568 | 10: 0.16792963 | 11: 0.16650155 | 12: 0.17860866
13: 0.24703386 | 14: 0.17314321 | 15: 0.17790862 | 16: 0.14349050 | 17: 0.19861099 | 18: 0.22179585
19: 0.25014590 | 20: 0.19186367 | 21: 0.20709049 | 22: 0.22461699 | 23: 0.20941889 | 24: 0.20251317
25: 0.19734152 | 26: 0.12080955 | 27: 0.15094576 | 28: 0.12359245 | 29: 0.20994047 | 30: 0.20793625
31: 0.16568922 | 32: 0.07936047 | 33: 0.23366203 | 34: 0.18522035 | 35: 0.21760164 | 36: 0.12062719
37: 0.15700110 | 38: 0.15548727 | 39: 0.16454941
DIJAGNOSTIKA: 5 treninga nema potvrđen gradijentni uslov; njihovi rezultati ne potvrđuju dostizanje minimuma.
DIJAGNOSTIKA: Koherentni model nije dao niži test NLL od dijagnostičke kontrole bez koherencije.

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 0.03
Trening prelaza: 1785 | regularizacija: 0.03
Iteracija 10/160 | proteklo 0.5 s | cilj 17.078150988 | Max|proj.grad| 4.164e-02
Iteracija 20/160 | proteklo 0.9 s | cilj 16.859368891 | Max|proj.grad| 2.082e-02
Iteracija 30/160 | proteklo 1.3 s | cilj 16.784721971 | Max|proj.grad| 4.073e-02
Iteracija 40/160 | proteklo 1.7 s | cilj 16.742442787 | Max|proj.grad| 4.427e-02
Iteracija 50/160 | proteklo 2.1 s | cilj 16.722654260 | Max|proj.grad| 5.818e-03
Iteracija 60/160 | proteklo 2.5 s | cilj 16.708908587 | Max|proj.grad| 9.751e-03
Iteracija 70/160 | proteklo 2.9 s | cilj 16.703834851 | Max|proj.grad| 5.389e-03
Iteracija 80/160 | proteklo 3.3 s | cilj 16.698825496 | Max|proj.grad| 1.319e-02
Iteracija 90/160 | proteklo 3.7 s | cilj 16.693829327 | Max|proj.grad| 4.064e-03
Iteracija 100/160 | proteklo 4.1 s | cilj 16.690028099 | Max|proj.grad| 6.770e-03
Iteracija 110/160 | proteklo 4.5 s | cilj 16.687769484 | Max|proj.grad| 2.496e-03
Iteracija 120/160 | proteklo 4.9 s | cilj 16.685000127 | Max|proj.grad| 5.629e-03
Iteracija 130/160 | proteklo 5.3 s | cilj 16.682594956 | Max|proj.grad| 7.530e-03
Iteracija 140/160 | proteklo 5.7 s | cilj 16.680253753 | Max|proj.grad| 2.128e-03
Iteracija 150/160 | proteklo 6.2 s | cilj 16.679180885 | Max|proj.grad| 3.692e-03
Iteracija 160/160 | proteklo 6.6 s | cilj 16.678716161 | Max|proj.grad| 2.045e-03
Zadržani cilj: 16.678716161
Max|proj.grad|: 2.045425e-03
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 16.827852019

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 0.03
Trening prelaza: 2083 | regularizacija: 0.03
Iteracija 10/160 | proteklo 0.5 s | cilj 16.699408217 | Max|proj.grad| 4.074e-03
Iteracija 20/160 | proteklo 1.0 s | cilj 16.697217429 | Max|proj.grad| 2.969e-03
Iteracija 30/160 | proteklo 1.5 s | cilj 16.696120362 | Max|proj.grad| 1.751e-03
Iteracija 40/160 | proteklo 2.0 s | cilj 16.695239978 | Max|proj.grad| 1.202e-02
Iteracija 50/160 | proteklo 2.5 s | cilj 16.694873198 | Max|proj.grad| 2.391e-03
Iteracija 60/160 | proteklo 3.0 s | cilj 16.694669555 | Max|proj.grad| 1.991e-03
Iteracija 70/160 | proteklo 3.4 s | cilj 16.694404602 | Max|proj.grad| 2.235e-03
Iteracija 80/160 | proteklo 3.9 s | cilj 16.694110073 | Max|proj.grad| 1.679e-03
Iteracija 90/160 | proteklo 4.4 s | cilj 16.693418591 | Max|proj.grad| 2.351e-03
Iteracija 100/160 | proteklo 4.9 s | cilj 16.692829926 | Max|proj.grad| 8.591e-04
Iteracija 110/160 | proteklo 5.5 s | cilj 16.692135664 | Max|proj.grad| 3.758e-03
Iteracija 120/160 | proteklo 6.0 s | cilj 16.690312169 | Max|proj.grad| 3.163e-03
Iteracija 130/160 | proteklo 6.5 s | cilj 16.688683585 | Max|proj.grad| 1.731e-03
Iteracija 140/160 | proteklo 7.0 s | cilj 16.687417583 | Max|proj.grad| 3.507e-03
Iteracija 150/160 | proteklo 7.5 s | cilj 16.686849204 | Max|proj.grad| 2.772e-03
Iteracija 160/160 | proteklo 8.0 s | cilj 16.686548862 | Max|proj.grad| 8.825e-04
Zadržani cilj: 16.686548862
Max|proj.grad|: 8.824830e-04
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.
Validacija 2 NLL: 16.799094968
Objedinjeni validacioni NLL za ridge 0.03: 16.813497659

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 0.3
Trening prelaza: 1785 | regularizacija: 0.3
Iteracija 10/160 | proteklo 0.4 s | cilj 17.129948725 | Max|proj.grad| 5.141e-02
Iteracija 20/160 | proteklo 0.8 s | cilj 16.966238847 | Max|proj.grad| 1.512e-02
Iteracija 30/160 | proteklo 1.2 s | cilj 16.911544998 | Max|proj.grad| 1.250e-02
Iteracija 40/160 | proteklo 1.7 s | cilj 16.876285985 | Max|proj.grad| 1.112e-02
Iteracija 50/160 | proteklo 2.1 s | cilj 16.862692241 | Max|proj.grad| 4.087e-03
Iteracija 60/160 | proteklo 2.6 s | cilj 16.855090579 | Max|proj.grad| 1.716e-02
Iteracija 70/160 | proteklo 3.0 s | cilj 16.845732773 | Max|proj.grad| 9.981e-03
Iteracija 80/160 | proteklo 3.4 s | cilj 16.837115925 | Max|proj.grad| 3.989e-03
Iteracija 90/160 | proteklo 3.8 s | cilj 16.832328018 | Max|proj.grad| 6.633e-03
Iteracija 100/160 | proteklo 4.3 s | cilj 16.828865527 | Max|proj.grad| 2.776e-03
Iteracija 110/160 | proteklo 4.7 s | cilj 16.827721684 | Max|proj.grad| 1.926e-03
Iteracija 120/160 | proteklo 5.1 s | cilj 16.827279952 | Max|proj.grad| 6.341e-04
Iteracija 130/160 | proteklo 5.6 s | cilj 16.827152233 | Max|proj.grad| 1.619e-03
Iteracija 140/160 | proteklo 6.0 s | cilj 16.827109750 | Max|proj.grad| 3.252e-04
Iteracija 150/160 | proteklo 6.5 s | cilj 16.827094793 | Max|proj.grad| 3.648e-04
Iteracija 160/160 | proteklo 6.9 s | cilj 16.827087855 | Max|proj.grad| 2.437e-04
Zadržani cilj: 16.827087855
Max|proj.grad|: 2.436776e-04
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 16.681889540

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 0.3
Trening prelaza: 2083 | regularizacija: 0.3
Iteracija 10/160 | proteklo 0.5 s | cilj 16.818792309 | Max|proj.grad| 3.745e-03
Iteracija 20/160 | proteklo 1.1 s | cilj 16.817394751 | Max|proj.grad| 8.045e-03
Iteracija 30/160 | proteklo 1.6 s | cilj 16.817174051 | Max|proj.grad| 5.205e-04
Iteracija 40/160 | proteklo 2.0 s | cilj 16.817040544 | Max|proj.grad| 1.618e-03
Iteracija 50/160 | proteklo 2.5 s | cilj 16.816884205 | Max|proj.grad| 1.280e-03
Iteracija 60/160 | proteklo 3.0 s | cilj 16.816758539 | Max|proj.grad| 1.380e-03
Iteracija 70/160 | proteklo 3.5 s | cilj 16.816379817 | Max|proj.grad| 2.351e-03
Iteracija 80/160 | proteklo 4.0 s | cilj 16.815707192 | Max|proj.grad| 4.674e-03
Iteracija 90/160 | proteklo 4.5 s | cilj 16.815463479 | Max|proj.grad| 2.458e-03
Iteracija 100/160 | proteklo 5.0 s | cilj 16.815400384 | Max|proj.grad| 2.587e-04
Iteracija 110/160 | proteklo 5.5 s | cilj 16.815382124 | Max|proj.grad| 1.571e-03
Iteracija 120/160 | proteklo 6.0 s | cilj 16.815374006 | Max|proj.grad| 1.790e-04
Iteracija 130/160 | proteklo 6.5 s | cilj 16.815370316 | Max|proj.grad| 1.326e-04
Iteracija 140/160 | proteklo 7.0 s | cilj 16.815368994 | Max|proj.grad| 4.079e-05
Iteracija 150/160 | proteklo 7.4 s | cilj 16.815368363 | Max|proj.grad| 1.110e-04
Iteracija 160/160 | proteklo 7.9 s | cilj 16.815368039 | Max|proj.grad| 3.985e-05
Zadržani cilj: 16.815368039
Max|proj.grad|: 3.985188e-05
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Validacija 2 NLL: 16.842582521
Objedinjeni validacioni NLL za ridge 0.3: 16.762100994

loto7_4710_k86_loto_2977.csv: izabrana regularizacija 0.3
Validacioni NLL: 16.762100994

loto7_4710_k86_loto_2977.csv / model za odvojeni test
Trening prelaza: 2380 | regularizacija: 0.3
Iteracija 10/160 | proteklo 0.6 s | cilj 16.830878486 | Max|proj.grad| 5.077e-03
Iteracija 20/160 | proteklo 1.1 s | cilj 16.829951698 | Max|proj.grad| 1.297e-03
Iteracija 30/160 | proteklo 1.6 s | cilj 16.829472625 | Max|proj.grad| 8.417e-04
Iteracija 40/160 | proteklo 2.1 s | cilj 16.829036992 | Max|proj.grad| 4.089e-03
Iteracija 50/160 | proteklo 2.7 s | cilj 16.827877169 | Max|proj.grad| 2.634e-03
Iteracija 60/160 | proteklo 3.2 s | cilj 16.827287589 | Max|proj.grad| 7.665e-04
Iteracija 70/160 | proteklo 3.7 s | cilj 16.827057335 | Max|proj.grad| 5.217e-04
Iteracija 80/160 | proteklo 4.2 s | cilj 16.826964478 | Max|proj.grad| 3.641e-04
Iteracija 90/160 | proteklo 4.8 s | cilj 16.826934057 | Max|proj.grad| 2.214e-04
Iteracija 100/160 | proteklo 5.3 s | cilj 16.826923922 | Max|proj.grad| 7.592e-05
Iteracija 110/160 | proteklo 5.8 s | cilj 16.826919394 | Max|proj.grad| 7.703e-05
Iteracija 120/160 | proteklo 6.4 s | cilj 16.826917467 | Max|proj.grad| 6.633e-05
Iteracija 130/160 | proteklo 7.0 s | cilj 16.826917036 | Max|proj.grad| 3.976e-05
Iteracija 140/160 | proteklo 7.5 s | cilj 16.826916897 | Max|proj.grad| 2.524e-05
Iteracija 150/160 | proteklo 8.0 s | cilj 16.826916854 | Max|proj.grad| 6.659e-06
Iteracija 160/160 | proteklo 8.6 s | cilj 16.826916844 | Max|proj.grad| 5.070e-06
Zadržani cilj: 16.826916844
Max|proj.grad|: 5.069826e-06
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv: odvojeni test je obrađen.
Test NLL: 16.867875627
Kontrole menjaju isto kvantno kolo i nisu posebno trenirani konkurenti.
Test kontrole ne učestvuju u izboru regularizacije.

loto7_4710_k86_loto_2977.csv / završno učenje na svim prelazima
Trening prelaza: 2976 | regularizacija: 0.3
Iteracija 10/160 | proteklo 0.8 s | cilj 16.855014030 | Max|proj.grad| 2.038e-03
Iteracija 20/160 | proteklo 1.5 s | cilj 16.853867068 | Max|proj.grad| 8.125e-04
Iteracija 30/160 | proteklo 2.2 s | cilj 16.853578943 | Max|proj.grad| 6.526e-04
Iteracija 40/160 | proteklo 2.8 s | cilj 16.853504048 | Max|proj.grad| 4.444e-04
Iteracija 50/160 | proteklo 3.5 s | cilj 16.853469569 | Max|proj.grad| 2.041e-04
Iteracija 60/160 | proteklo 4.1 s | cilj 16.853455636 | Max|proj.grad| 1.389e-04
Iteracija 70/160 | proteklo 4.7 s | cilj 16.853450119 | Max|proj.grad| 5.965e-05
Iteracija 80/160 | proteklo 5.5 s | cilj 16.853447822 | Max|proj.grad| 1.260e-04
Iteracija 90/160 | proteklo 6.1 s | cilj 16.853446980 | Max|proj.grad| 3.766e-04
Iteracija 100/160 | proteklo 6.7 s | cilj 16.853446700 | Max|proj.grad| 5.227e-05
Iteracija 110/160 | proteklo 7.4 s | cilj 16.853446622 | Max|proj.grad| 9.159e-06
Iteracija 120/160 | proteklo 8.0 s | cilj 16.853446597 | Max|proj.grad| 8.751e-06
Iteracija 130/160 | proteklo 8.7 s | cilj 16.853446593 | Max|proj.grad| 3.127e-06
Zadržani cilj: 16.853446592
Max|proj.grad|: 6.351746e-07
Status optimizatora: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL

loto7_4710_k86_loto_2977.csv
NEXT: 1 x 9 y 19 z 38
Validacioni NLL: 16.762100994
Odvojeni test NLL: 16.867875627
Test NLL bez faza dve akcione kapije: 18.792304919
Test NLL sa uklonjenom koherencijom putanja: 16.610914830
Modelna verovatnoća NEXT: 5.66590321905e-07
Normalizacioni odnos: 1.000000000000
Greška dve normalizacije: 1.776e-15
Greška Qiskit amplituda: 1.403e-16
Greška eksplicitnog zbira putanja: 1.388e-16
Razlika koherentnog i nekoherentnog zbira za NEXT brojeve: 1.810471e-02
Dijagnostička osetljivost na istoriju: 1.816490e-01
Zbir marginalnih verovatnoća uključivanja: 7.000000000000
Završni Max|proj.grad|: 6.351746e-07
Test: broj kola prema broju pogođenih brojeva
0: 132 | 1: 232 | 2: 180 | 3: 43 | 4: 9 | 5: 0 | 6: 0 | 7: 0
Prosečan broj pogodaka na testu: 1.270134
Marginalna verovatnoća uključivanja svakog broja:
01: 0.21975207 | 02: 0.16208023 | 03: 0.17081969 | 04: 0.12802765 | 05: 0.16544320 | 06: 0.21211628
07: 0.19382373 | 08: 0.26462779 | 09: 0.22131723 | 10: 0.11783612 | 11: 0.15435957 | 12: 0.18724579
13: 0.21403491 | 14: 0.17952477 | 15: 0.15428679 | 16: 0.15159641 | 17: 0.22942524 | 18: 0.21082470
19: 0.25092151 | 20: 0.12189082 | 21: 0.21577606 | 22: 0.15984203 | 23: 0.22183408 | 24: 0.19937306
25: 0.19331338 | 26: 0.18753202 | 27: 0.18176262 | 28: 0.14534497 | 29: 0.16204660 | 30: 0.17335522
31: 0.13757836 | 32: 0.19970772 | 33: 0.16225435 | 34: 0.15857228 | 35: 0.15474583 | 36: 0.10021615
37: 0.15070267 | 38: 0.21983931 | 39: 0.16624878
DIJAGNOSTIKA: 3 treninga nema potvrđen gradijentni uslov; njihovi rezultati ne potvrđuju dostizanje minimuma.
DIJAGNOSTIKA: Koherentni model nije dao niži test NLL od dijagnostičke kontrole bez koherencije.

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 0.03
Trening prelaza: 1039 | regularizacija: 0.03
Iteracija 10/160 | proteklo 0.3 s | cilj 17.017410002 | Max|proj.grad| 4.844e-02
Iteracija 20/160 | proteklo 0.6 s | cilj 16.795176177 | Max|proj.grad| 3.078e-02
Iteracija 30/160 | proteklo 0.8 s | cilj 16.719995542 | Max|proj.grad| 1.264e-02
Iteracija 40/160 | proteklo 1.1 s | cilj 16.693533530 | Max|proj.grad| 1.342e-02
Iteracija 50/160 | proteklo 1.3 s | cilj 16.678447471 | Max|proj.grad| 9.749e-03
Iteracija 60/160 | proteklo 1.6 s | cilj 16.658826235 | Max|proj.grad| 3.665e-02
Iteracija 70/160 | proteklo 1.9 s | cilj 16.645182218 | Max|proj.grad| 7.003e-03
Iteracija 80/160 | proteklo 2.2 s | cilj 16.636329793 | Max|proj.grad| 1.105e-02
Iteracija 90/160 | proteklo 2.4 s | cilj 16.633782916 | Max|proj.grad| 9.767e-03
Iteracija 100/160 | proteklo 2.7 s | cilj 16.629969889 | Max|proj.grad| 1.348e-02
Iteracija 110/160 | proteklo 3.0 s | cilj 16.626596115 | Max|proj.grad| 2.010e-03
Iteracija 120/160 | proteklo 3.2 s | cilj 16.624799187 | Max|proj.grad| 1.590e-03
Iteracija 130/160 | proteklo 3.5 s | cilj 16.623574602 | Max|proj.grad| 3.482e-03
Iteracija 140/160 | proteklo 3.8 s | cilj 16.621618982 | Max|proj.grad| 5.858e-03
Iteracija 150/160 | proteklo 4.1 s | cilj 16.619133647 | Max|proj.grad| 9.610e-03
Iteracija 160/160 | proteklo 4.3 s | cilj 16.614695906 | Max|proj.grad| 6.998e-03
Zadržani cilj: 16.614695906
Max|proj.grad|: 6.998472e-03
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 16.967022300

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 0.03
Trening prelaza: 1212 | regularizacija: 0.03
Iteracija 10/160 | proteklo 0.3 s | cilj 16.652309855 | Max|proj.grad| 8.274e-03
Iteracija 20/160 | proteklo 0.6 s | cilj 16.634420573 | Max|proj.grad| 7.796e-03
Iteracija 30/160 | proteklo 0.9 s | cilj 16.625582807 | Max|proj.grad| 3.550e-03
Iteracija 40/160 | proteklo 1.2 s | cilj 16.621095817 | Max|proj.grad| 3.053e-03
Iteracija 50/160 | proteklo 1.5 s | cilj 16.614747150 | Max|proj.grad| 9.092e-03
Iteracija 60/160 | proteklo 1.8 s | cilj 16.606270582 | Max|proj.grad| 7.341e-03
Iteracija 70/160 | proteklo 2.1 s | cilj 16.601807690 | Max|proj.grad| 5.559e-03
Iteracija 80/160 | proteklo 2.4 s | cilj 16.599821963 | Max|proj.grad| 1.562e-03
Iteracija 90/160 | proteklo 2.7 s | cilj 16.598882631 | Max|proj.grad| 2.278e-03
Iteracija 100/160 | proteklo 3.0 s | cilj 16.598231622 | Max|proj.grad| 1.353e-03
Iteracija 110/160 | proteklo 3.3 s | cilj 16.597248097 | Max|proj.grad| 2.169e-03
Iteracija 120/160 | proteklo 3.6 s | cilj 16.596314801 | Max|proj.grad| 4.355e-03
Iteracija 130/160 | proteklo 3.9 s | cilj 16.595133866 | Max|proj.grad| 4.099e-03
Iteracija 140/160 | proteklo 4.2 s | cilj 16.593511325 | Max|proj.grad| 1.787e-03
Iteracija 150/160 | proteklo 4.4 s | cilj 16.592163530 | Max|proj.grad| 2.947e-03
Iteracija 160/160 | proteklo 4.7 s | cilj 16.591472538 | Max|proj.grad| 1.434e-03
Zadržani cilj: 16.591472538
Max|proj.grad|: 1.433525e-03
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.
Validacija 2 NLL: 16.972685745
Objedinjeni validacioni NLL za ridge 0.03: 16.969854022

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 0.3
Trening prelaza: 1039 | regularizacija: 0.3
Iteracija 10/160 | proteklo 0.3 s | cilj 17.084820751 | Max|proj.grad| 1.738e-01
Iteracija 20/160 | proteklo 0.5 s | cilj 16.877474425 | Max|proj.grad| 9.567e-03
Iteracija 30/160 | proteklo 0.8 s | cilj 16.822676974 | Max|proj.grad| 2.837e-02
Iteracija 40/160 | proteklo 1.1 s | cilj 16.807535875 | Max|proj.grad| 4.730e-02
Iteracija 50/160 | proteklo 1.4 s | cilj 16.800550266 | Max|proj.grad| 3.137e-03
Iteracija 60/160 | proteklo 1.6 s | cilj 16.795431943 | Max|proj.grad| 2.646e-03
Iteracija 70/160 | proteklo 1.9 s | cilj 16.792260443 | Max|proj.grad| 5.439e-03
Iteracija 80/160 | proteklo 2.2 s | cilj 16.790410772 | Max|proj.grad| 6.099e-03
Iteracija 90/160 | proteklo 2.4 s | cilj 16.789821750 | Max|proj.grad| 1.616e-03
Iteracija 100/160 | proteklo 2.8 s | cilj 16.789352088 | Max|proj.grad| 1.068e-03
Iteracija 110/160 | proteklo 3.0 s | cilj 16.788460530 | Max|proj.grad| 6.840e-03
Iteracija 120/160 | proteklo 3.4 s | cilj 16.778722937 | Max|proj.grad| 7.493e-03
Iteracija 130/160 | proteklo 3.6 s | cilj 16.775980713 | Max|proj.grad| 2.816e-03
Iteracija 140/160 | proteklo 3.9 s | cilj 16.774828028 | Max|proj.grad| 3.672e-03
Iteracija 150/160 | proteklo 4.3 s | cilj 16.774043482 | Max|proj.grad| 5.545e-03
Iteracija 160/160 | proteklo 4.5 s | cilj 16.773034725 | Max|proj.grad| 3.348e-03
Zadržani cilj: 16.773034725
Max|proj.grad|: 3.348012e-03
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.
Validacija 1 NLL: 16.962398475

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 0.3
Trening prelaza: 1212 | regularizacija: 0.3
Iteracija 10/160 | proteklo 0.3 s | cilj 16.805081402 | Max|proj.grad| 3.620e-03
Iteracija 20/160 | proteklo 0.7 s | cilj 16.801134397 | Max|proj.grad| 3.226e-03
Iteracija 30/160 | proteklo 1.1 s | cilj 16.800109047 | Max|proj.grad| 4.631e-03
Iteracija 40/160 | proteklo 1.4 s | cilj 16.799865837 | Max|proj.grad| 9.453e-04
Iteracija 50/160 | proteklo 1.7 s | cilj 16.799751401 | Max|proj.grad| 4.799e-04
Iteracija 60/160 | proteklo 2.0 s | cilj 16.799627236 | Max|proj.grad| 1.257e-03
Iteracija 70/160 | proteklo 2.3 s | cilj 16.799415873 | Max|proj.grad| 9.742e-04
Iteracija 80/160 | proteklo 2.6 s | cilj 16.799202874 | Max|proj.grad| 6.688e-04
Iteracija 90/160 | proteklo 3.0 s | cilj 16.799064089 | Max|proj.grad| 2.922e-03
Iteracija 100/160 | proteklo 3.2 s | cilj 16.798957197 | Max|proj.grad| 9.982e-04
Iteracija 110/160 | proteklo 3.5 s | cilj 16.798913403 | Max|proj.grad| 1.892e-04
Iteracija 120/160 | proteklo 3.8 s | cilj 16.798895770 | Max|proj.grad| 2.543e-04
Iteracija 130/160 | proteklo 4.1 s | cilj 16.798889604 | Max|proj.grad| 1.982e-04
Iteracija 140/160 | proteklo 4.5 s | cilj 16.798885957 | Max|proj.grad| 8.812e-05
Iteracija 150/160 | proteklo 4.8 s | cilj 16.798884035 | Max|proj.grad| 9.189e-05
Iteracija 160/160 | proteklo 5.1 s | cilj 16.798883001 | Max|proj.grad| 1.179e-04
Zadržani cilj: 16.798883001
Max|proj.grad|: 1.178675e-04
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.
Validacija 2 NLL: 17.009082124
Objedinjeni validacioni NLL za ridge 0.3: 16.985740299

loto7_4710_k86_loto_plus_1733.csv: izabrana regularizacija 0.03
Validacioni NLL: 16.969854022

loto7_4710_k86_loto_plus_1733.csv / model za odvojeni test
Trening prelaza: 1385 | regularizacija: 0.03
Iteracija 10/160 | proteklo 0.4 s | cilj 16.633493179 | Max|proj.grad| 2.761e-03
Iteracija 20/160 | proteklo 0.7 s | cilj 16.630193007 | Max|proj.grad| 3.173e-03
Iteracija 30/160 | proteklo 1.1 s | cilj 16.628831439 | Max|proj.grad| 1.435e-03
Iteracija 40/160 | proteklo 1.4 s | cilj 16.628038967 | Max|proj.grad| 1.793e-03
Iteracija 50/160 | proteklo 1.7 s | cilj 16.627516970 | Max|proj.grad| 2.787e-03
Iteracija 60/160 | proteklo 2.1 s | cilj 16.627045572 | Max|proj.grad| 9.251e-04
Iteracija 70/160 | proteklo 2.4 s | cilj 16.626402581 | Max|proj.grad| 1.134e-03
Iteracija 80/160 | proteklo 2.7 s | cilj 16.625832283 | Max|proj.grad| 3.015e-03
Iteracija 90/160 | proteklo 3.0 s | cilj 16.625455726 | Max|proj.grad| 1.187e-03
Iteracija 100/160 | proteklo 3.3 s | cilj 16.625277053 | Max|proj.grad| 6.764e-04
Iteracija 110/160 | proteklo 3.7 s | cilj 16.625181061 | Max|proj.grad| 2.861e-04
Iteracija 120/160 | proteklo 4.0 s | cilj 16.625142144 | Max|proj.grad| 2.729e-04
Iteracija 130/160 | proteklo 4.3 s | cilj 16.625126800 | Max|proj.grad| 2.344e-04
Iteracija 140/160 | proteklo 4.7 s | cilj 16.625116845 | Max|proj.grad| 4.616e-04
Iteracija 150/160 | proteklo 5.0 s | cilj 16.625110557 | Max|proj.grad| 8.425e-05
Iteracija 160/160 | proteklo 5.3 s | cilj 16.625106628 | Max|proj.grad| 5.637e-05
Zadržani cilj: 16.625106628
Max|proj.grad|: 5.637332e-05
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv: odvojeni test je obrađen.
Test NLL: 16.866642408
Kontrole menjaju isto kvantno kolo i nisu posebno trenirani konkurenti.
Test kontrole ne učestvuju u izboru regularizacije.

loto7_4710_k86_loto_plus_1733.csv / završno učenje na svim prelazima
Trening prelaza: 1732 | regularizacija: 0.03
Iteracija 10/160 | proteklo 0.4 s | cilj 16.673333375 | Max|proj.grad| 3.469e-03
Iteracija 20/160 | proteklo 0.8 s | cilj 16.670472179 | Max|proj.grad| 3.067e-03
Iteracija 30/160 | proteklo 1.2 s | cilj 16.669049960 | Max|proj.grad| 2.013e-03
Iteracija 40/160 | proteklo 1.6 s | cilj 16.666905258 | Max|proj.grad| 2.924e-03
Iteracija 50/160 | proteklo 2.1 s | cilj 16.665231161 | Max|proj.grad| 1.461e-03
Iteracija 60/160 | proteklo 2.5 s | cilj 16.664112143 | Max|proj.grad| 2.523e-03
Iteracija 70/160 | proteklo 2.9 s | cilj 16.662905796 | Max|proj.grad| 7.402e-03
Iteracija 80/160 | proteklo 3.3 s | cilj 16.661165918 | Max|proj.grad| 1.038e-02
Iteracija 90/160 | proteklo 3.6 s | cilj 16.659729142 | Max|proj.grad| 2.345e-03
Iteracija 100/160 | proteklo 4.0 s | cilj 16.659087099 | Max|proj.grad| 8.851e-04
Iteracija 110/160 | proteklo 4.4 s | cilj 16.658672335 | Max|proj.grad| 3.189e-03
Iteracija 120/160 | proteklo 4.8 s | cilj 16.657830480 | Max|proj.grad| 2.154e-03
Iteracija 130/160 | proteklo 5.2 s | cilj 16.656295473 | Max|proj.grad| 3.038e-03
Iteracija 140/160 | proteklo 5.6 s | cilj 16.655390491 | Max|proj.grad| 9.173e-04
Iteracija 150/160 | proteklo 6.1 s | cilj 16.654963931 | Max|proj.grad| 7.250e-04
Iteracija 160/160 | proteklo 6.5 s | cilj 16.654670339 | Max|proj.grad| 2.790e-03
Zadržani cilj: 16.654670339
Max|proj.grad|: 2.789504e-03
Status optimizatora: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov konvergencije nije potvrđen.
Kandidat ima proverenu raspodelu; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv
NEXT: 6 x 20 y 32 z 36
Validacioni NLL: 16.969854022
Odvojeni test NLL: 16.866642408
Test NLL bez faza dve akcione kapije: 18.941324230
Test NLL sa uklonjenom koherencijom putanja: 16.615889833
Modelna verovatnoća NEXT: 8.82451533137e-07
Normalizacioni odnos: 1.000000000000
Greška dve normalizacije: 0.000e+00
Greška Qiskit amplituda: 1.671e-16
Greška eksplicitnog zbira putanja: 1.186e-16
Razlika koherentnog i nekoherentnog zbira za NEXT brojeve: 1.856757e-02
Dijagnostička osetljivost na istoriju: 1.943336e-01
Zbir marginalnih verovatnoća uključivanja: 7.000000000000
Završni Max|proj.grad|: 2.789504e-03
Test: broj kola prema broju pogođenih brojeva
0: 73 | 1: 133 | 2: 109 | 3: 30 | 4: 2 | 5: 0 | 6: 0 | 7: 0
Prosečan broj pogodaka na testu: 1.293948
Marginalna verovatnoća uključivanja svakog broja:
01: 0.15488145 | 02: 0.22834687 | 03: 0.12627675 | 04: 0.15505949 | 05: 0.16900342 | 06: 0.25826412
07: 0.22657199 | 08: 0.21375890 | 09: 0.13936406 | 10: 0.23343168 | 11: 0.15964116 | 12: 0.12401982
13: 0.08430574 | 14: 0.16423218 | 15: 0.17079478 | 16: 0.21048765 | 17: 0.14173950 | 18: 0.18400612
19: 0.12946779 | 20: 0.23210019 | 21: 0.13579833 | 22: 0.17948996 | 23: 0.14423618 | 24: 0.15468813
25: 0.10867471 | 26: 0.21226338 | 27: 0.14504784 | 28: 0.20538218 | 29: 0.17033219 | 30: 0.22028900
31: 0.23396682 | 32: 0.26079617 | 33: 0.16575930 | 34: 0.22595851 | 35: 0.22941713 | 36: 0.26567960
37: 0.15837422 | 38: 0.15296496 | 39: 0.12512773
DIJAGNOSTIKA: 6 treninga nema potvrđen gradijentni uslov; njihovi rezultati ne potvrđuju dostizanje minimuma.
DIJAGNOSTIKA: Koherentni model nije dao niži test NLL od dijagnostičke kontrole bez koherencije.

ZAVRŠNI NEXT REZULTATI
loto7_4710_k86.csv | NEXT: 5 13 18 19 22 33 35
loto7_4710_k86_loto_2977.csv | NEXT: 1 8 9 17 19 23 38
loto7_4710_k86_loto_plus_1733.csv | NEXT: 6 10 20 31 32 35 36
"""



"""
Zbir kompleksnih amplituda različitih putanja i njihova interferencija. 
Feynman rad iz 1948.. Ključ je sledeći: 
alternativne putanje daju kompleksne doprinose koji se sabiraju pre računanja verovatnoće. 
Faze određuju međusobno pojačavanje i poništavanje.

Za moj Loto to daje konkretnu istraživačku ideju:
- Putanje: različiti nizovi unutrašnjih stanja kvantnog modela, uslovljeni prethodnim kolima.
- Učene faze: istorijski prelazi određuju parametre faznih kapija.
- Interferencija: Qiskit kolo ponovo spaja alternative, tako da faze menjaju izlazne verovatnoće.
- Izlaz: normalizovana raspodela nad legalnim sedmorkama, iz koje se bira NEXT.

Poređenje istog modela sa učenim fazama i sa uklonjenim faznim razlikama. 
Ako interferencija ne poboljša rezultat na odvojenom hronološkom testu, nemam dokaz da je ova ideja korisna za predikciju.

V1 sa Qiskit kolom koje sabira putanje kroz učene faze i interferenciju. 
Uključiću poređenje sa istim modelom bez faznih razlika, 
da test pokaže da li interferencija doprinosi predikciji. 

V1: kompleksne amplitude putanja, učene faze i interferenciju u Qiskit kolu. 
Sedam registara deli isto kolo; raspodela sedmorki dobija se uslovljavanjem na sedam različitih brojeva. 
Registri nisu međusobno kvantno spregnuti.
"""
