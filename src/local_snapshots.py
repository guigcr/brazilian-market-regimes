"""Read private research inputs while preserving the archived source format."""

import ast
import os
from functools import lru_cache
from pathlib import Path

import nbformat


_ARCHIVED_NOTEBOOKS = {
    "Ibovespa_Risk_Regimes.ipynb": "Ibovespa_Regimes_de_Risco.ipynb",
    "Cross_Market_Features_and_Meta_Labeling.ipynb": "Ibovespa_Mercados_e_MetaLabeling.ipynb",
}
_ARCHIVED_VARIABLES = {
    "SNAPSHOT": "BASE",
    "SNAPSHOTS": "BASES",
    "FROZEN_INTRADAY": "INTRADIA_CONGELADO",
}
_ARCHIVED_KEYS = {
    "dolar": "usd", "petroleo": "oil", "sinal": "signal",
    "modelo_ubj": "model_ubj", "treino_n": "training_count",
    "treino_primeiro": "first_training_date", "treino_ultimo": "last_training_date",
    "media": "mean", "variancia": "variance", "inicial": "initial",
    "transicao": "transition", "variancia_inicial": "variance_initial",
    "custo_lado": "cost_per_side", "margem": "margin",
}
_ARCHIVED_FEATURE_PARTS = {
    "dist_media": "dist_mean", "intradia": "intraday", "amplitude": "range",
    "dia": "day", "hmm_vol_alta": "hmm_high_volatility",
    "garch_vol_prox": "garch_next_volatility", "dolar": "usd", "petroleo": "oil",
}


def private_root():
    path = os.environ.get("IBOV_LOCAL_RESEARCH") or os.environ.get("IBOV_PESQUISA_LOCAL")
    if not path:
        raise ValueError("Set IBOV_LOCAL_RESEARCH to the original private research directory.")
    root = Path(path).expanduser().resolve()
    if root == Path(__file__).resolve().parents[1]:
        raise ValueError("Point to the original research directory outside this checkout.")
    return root


def raw_ibovespa_path():
    return private_root() / "dados/ibovespa_bruto.csv"


def _public_feature_name(name):
    for original, translated in _ARCHIVED_FEATURE_PARTS.items():
        if name == original or name.startswith(original + "_"):
            return translated + name[len(original):]
    return name


def _public_snapshot(values):
    return {
        _ARCHIVED_KEYS.get(key, key): _public_snapshot(value) if isinstance(value, dict) else value
        for key, value in values.items()
    }


@lru_cache
def load_local_snapshot(notebook, name):
    path = private_root() / _ARCHIVED_NOTEBOOKS.get(notebook, notebook)
    archived_name = _ARCHIVED_VARIABLES[name]
    for cell in nbformat.read(path, as_version=4).cells:
        if cell.cell_type != "code" or not any(
            line.startswith(f"{archived_name} =") for line in cell.source.splitlines()
        ):
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == archived_name for target in node.targets
            ):
                snapshot = _public_snapshot(ast.literal_eval(node.value))
                if name == "FROZEN_INTRADAY":
                    snapshot["features"] = [_public_feature_name(feature) for feature in snapshot["features"]]
                return snapshot
    raise ValueError(f"Snapshot {name} is missing from {path}.")
