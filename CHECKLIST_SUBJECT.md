# Checklist de Conformidade do Subject

Este documento resume, item por item, o estado de conformidade do projeto com o subject e com as validacoes executadas no ambiente.

## Como ler

- Status OK: item atendido e validado.
- Status ATENCAO: item depende de contexto externo (exemplo: disponibilidade de modelo/rede) ou validacao adicional manual.

## Checklist

| Item | Status | Evidencia |
|---|---|---|
| Projeto executa por linha de comando | OK | Execucao via [src/cli.py](src/cli.py) e [src/__main__.py](src/__main__.py) |
| Regras de Makefile disponiveis | OK | Targets em [Makefile](Makefile): install, run, debug, clean, lint, grade |
| Entrada por JSON de funcoes | OK | Leitura e validacao em [src/io_utils.py](src/io_utils.py) |
| Entrada por JSON de prompts | OK | Leitura e validacao em [src/io_utils.py](src/io_utils.py) |
| Saida em JSON estruturado | OK | Escrita em [src/io_utils.py](src/io_utils.py) para [data/output/function_calling_results.json](data/output/function_calling_results.json) |
| Campos de saida no formato esperado | OK | Modelo em [src/models.py](src/models.py), classe FunctionCallResult |
| Parametros coerentes com schema | OK | Coercao em [src/decoder.py](src/decoder.py) |
| Selecao de funcao com suporte de LLM | OK | Integracao em [src/llm_client.py](src/llm_client.py) |
| Fluxo robusto para erros de input | OK | Excecoes amigaveis em [src/io_utils.py](src/io_utils.py) e [src/errors.py](src/errors.py) |
| Casos de regex do conjunto publico | OK | Tratamento em [src/extractor.py](src/extractor.py) |
| Grade publico da moulinette | OK | Resultado confirmado: 11/11 |
| Testes unitarios locais | OK | Suite em [tests/test_decoder.py](tests/test_decoder.py) e [tests/test_cleaning.py](tests/test_cleaning.py) |
| Conformidade flake8 | OK | Validado localmente sem erros |
| Conformidade mypy | OK | Validado localmente sem erros |
| Uso de numpy no projeto | OK | Utilizado em [src/llm_client.py](src/llm_client.py) |
| Dependencias pesadas do SDK | ATENCAO | Podem exigir download grande dependendo do ambiente (torch e pacotes relacionados) |

## Comandos de validacao

```sh
uv sync
make run
make grade
/home/danicort/cPython/callmemaybe/.venv/bin/flake8 .
/home/danicort/cPython/callmemaybe/.venv/bin/mypy . --exclude '^llm_sdk/' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
/home/danicort/cPython/callmemaybe/.venv/bin/pytest -q
```

## Checklist rapido para entrega

- Gerar saida em [data/output/function_calling_results.json](data/output/function_calling_results.json).
- Rodar grade com 11/11.
- Garantir flake8, mypy e pytest verdes.
- Entregar sem arquivos temporarios desnecessarios.
