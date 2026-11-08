"""Lê dados e parâmetros de notebooks arquivados em uma pasta local separada."""

import ast
import os
from functools import lru_cache
from pathlib import Path

import nbformat


def raiz_privada():
    caminho = os.environ.get("IBOV_PESQUISA_LOCAL")
    if not caminho:
        raise ValueError("Defina IBOV_PESQUISA_LOCAL com a pasta privada do projeto original.")
    raiz = Path(caminho).expanduser().resolve()
    if raiz == Path(__file__).resolve().parents[1]:
        raise ValueError("Escolha a pasta de pesquisa original, fora da edição GitHub.")
    return raiz


@lru_cache
def ler_retrato_local(notebook, nome):
    caminho = raiz_privada() / notebook
    for celula in nbformat.read(caminho, as_version=4).cells:
        if celula.cell_type != "code" or not any(
            linha.startswith(f"{nome} =") for linha in celula.source.splitlines()
        ):
            continue
        for no in ast.parse(celula.source).body:
            if isinstance(no, ast.Assign) and any(
                isinstance(alvo, ast.Name) and alvo.id == nome for alvo in no.targets
            ):
                return ast.literal_eval(no.value)
    raise ValueError(f"Retrato {nome} ausente em {caminho}.")
