# ByteDrive Development Guidelines

**English** · [简体中文](开发规范.md)

These rules are mandatory for every change to this repository. Review this document before writing code.

## Contents

1. [Core principles](#1-core-principles)
2. [Directory responsibilities](#2-directory-responsibilities)
3. [Per-file documentation](#3-per-file-documentation)
4. [Chinese comment convention](#4-chinese-comment-convention)
5. [Documentation index](#5-documentation-index)
6. [Centralized configuration](#6-centralized-configuration)
7. [Validation rules](#7-validation-rules)
8. [Code structure and concision](#8-code-structure-and-concision)
9. [Avoid unnecessary loops](#9-avoid-unnecessary-loops)
10. [Pre-commit checklist](#10-pre-commit-checklist)

## 1. Core principles

| Principle | Rule |
| --- | --- |
| External configuration | Define every adjustable value only in `config/`. Implementation files read configuration and never redeclare defaults. |
| External validation | Put validation in the module's `checks/<module>_checks.py`. Keep implementation files focused on core logic. |
| Synchronized documentation | Every implementation file has a header; update `Doc/Index.md` whenever files are added or removed. |
| Concision | Prefer the shortest clear implementation that fully implements the behavior. Avoid duplication and ceremony. |
| Expression over loops | Prefer vectorization, comprehensions, and higher-order operations; use `for` when it is the clearer or necessary form. |
| Chinese source comments | Comments and docstrings are written in Chinese and explain *why*, not merely *what*. Identifiers remain English. |

## 2. Directory responsibilities

| Directory | Responsibility | Configurable values allowed? |
| --- | --- | --- |
| `config/` | All parameters, config loading, and config validation | **Yes — the only source** |
| `data/` | Data reading, preprocessing, datasets, and loaders | No; read config |
| `model/` | Network definitions | No |
| `train/` | Training, optimization, and evaluation loops | No |
| `clone_loop/` | Behavior-cloning inference/control closed loop | No |
| `vis/` | Visualization and log rendering | No |
| `Doc/` | Documentation and index | N/A |

If a value may change between experiments or environments, it belongs in `config/`. Shape derivations and mathematical constants such as `math.pi` are not configuration.

### 2.1 One folder per module

Each module owns a directory. Sibling implementations must not be flattened together:

```text
<package>/<module>/
├── __init__.py              # Header plus stable public API re-exports
├── <module>.py              # Core implementation
└── checks/                  # Optional when no precondition exists
    ├── __init__.py
    └── <module>_checks.py
```

- **Re-exports:** cross-module imports use `from <package>.<module> import X`; `__init__.py` keeps callers stable if implementation paths change.
- **Exceptions:** CLI `run.py` files remain at package roots. `config/` retains the flat structure described in section 6.1. Third-party `data/carla_data_collector/agents/` and the adapted official `scene_layout.py` do not follow this convention.

## 3. Per-file documentation

Every `.py` implementation file must begin with a module docstring in this form:

```python
"""<One-line responsibility: what this file does>

模块: train/loop.py
依赖: model.policy, data.loader, config
读取配置: train.lr, train.epochs, train.device
对外接口:
    - train_one_epoch(model, loader, cfg) -> float
    - evaluate(model, loader, cfg) -> dict
说明: <Optional key design decisions or cautions>
"""
```

The labels remain Chinese because source comments and docstrings follow the Chinese convention.

- `读取配置` must list every config key actually read by the file. Missing keys or stale entries are defects.
- `对外接口` lists public functions and classes only, one per line with a return type or description.
- A header that disagrees with the implementation is a code defect.
- Companion `checks/<module>_checks.py` files and `checks/__init__.py` are exempt from the module header. Their individual validation functions must still identify their validation target as required by section 7.2.
- A module `__init__.py` needs a header whose public API list matches its re-exports and the implementation's `__all__`.

## 4. Chinese comment convention

- Comments and docstrings are Chinese; variable, function, class, and module identifiers are English.
- Public functions require docstrings. Private or self-evident functions may omit them.

```python
def sample_actions(obs, policy, cfg):
    """根据观测采样动作。

    参数:
        obs:    形状 (B, C, H, W) 的图像观测张量
        policy: 已加载权重的策略网络
        cfg:    配置对象，读取 cfg.clone.temperature
    返回:
        (B, action_dim) 的动作张量
    """
```

- Inline comments appear only when the reason for an implementation choice is not obvious. Comments that restate code are prohibited.
- Use `# TODO(name): content` and `# FIXME(name): content` consistently.

## 5. Documentation index

`Doc/Index.md` is the single navigation entry for project documentation and source files.

- Update `Doc/Index.md` in the same commit whenever an implementation or documentation file is added or removed.
- Group entries by directory and format each line as `relative path — one-line responsibility`.
- The responsibility must match the first line of the file header so descriptions do not drift.
- English primary documents and their Chinese translations are both listed under documentation; translated source-file entries are not duplicated.

## 6. Centralized configuration

### 6.1 Structure

```text
config/
├── default.yaml      # Default values for every parameter; the only value source
├── schema.py         # Types and constraints
└── __init__.py       # load_config(): YAML -> schema -> validated config object
```

Parameter values live in `default.yaml`; types and constraints live in `schema.py`. Environment differences belong in `config/<env>.yaml` overrides, not implementation branches.

### 6.2 Non-negotiable rules

1. **Implementation files must not contain literal configuration.** Assignments such as `lr = 3e-4`, `batch_size = 32`, or `epochs = 100` are prohibited when they would change across experiments.
2. Implementation code receives a read-only `cfg` object or calls `load_config`. It does not mutate configuration.
3. A parameter has exactly one default occurrence in `default.yaml`; do not repeat fallback values in multiple files.

### 6.3 Examples

```python
# ✅ 正确：值来自 config
def build_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

# ❌ 错误：实现文件里二次配置
def build_optimizer(model):
    lr = 3e-4
    return torch.optim.AdamW(model.parameters(), lr=lr)
```

## 7. Validation rules

Keep only necessary checks and prevent validation from obscuring the implementation.

### 7.1 One adjacent validation file per implementation

Do not create a global validation monolith. Every `<module>/<module>.py` uses the adjacent `checks/<module>_checks.py` when runtime checks are needed.

```text
train/loop/
├── __init__.py
├── loop.py
└── checks/
    ├── __init__.py
    └── loop_checks.py
```

- The `checks/` directory shares the implementation's lifecycle. Delete it with the module. Omit it when no precondition exists.
- Each companion check file serves exactly one implementation file.
- Invoke checks once at an entry point or through a decorator; do not scatter assertions throughout core logic.

```python
from train.loop.checks.loop_checks import check_train_inputs

def train_one_epoch(model, loader, cfg):
    check_train_inputs(model, loader, cfg)
    ...
```

Config-value validation is the only exception: it remains in `config/schema.py`, where it runs once at load time. Inline assertions should be rare and precise.

### 7.2 Identify the validation target

Every check states which function, variable, or config field it validates:

```python
# 校验对象: cfg.train.lr —— 学习率必须为正
assert cfg.train.lr > 0, "train.lr 必须 > 0"

# 校验对象: 函数 sample_actions 的入参 obs —— 必须是 4 维图像张量
def _check_obs(obs):
    assert obs.ndim == 4, "obs 期望 (B,C,H,W) 四维"
```

Unidentified checks are noncompliant.

### 7.3 Necessity

- Do not repeat in runtime checks a constraint that `schema.py` can reject at load time.
- Validate each constraint once and near its source: configuration at load time, data and arguments at the implementation entry point.
- Do not add defensive branches for impossible conditions.

## 8. Code structure and concision

- **Single responsibility:** one function does one job; consider splitting functions longer than roughly 40 lines.
- **No duplication:** extract repeated logic on its second occurrence.
- **Early return:** use guard clauses to avoid deeply nested `if` blocks.
- **Names as documentation:** do not add a comment when a better name makes it unnecessary.
- **Delete dead code:** do not commit commented-out implementations; Git retains history.
- **Imports:** standard library, third party, then project imports, separated by blank lines.

## 9. Avoid unnecessary loops

When alternatives are equivalent and no less readable, prefer:

| Priority | Form | Use |
| --- | --- | --- |
| 1 | NumPy/PyTorch/pandas vectorization | Numeric batch operations |
| 2 | Comprehension or generator expression | Collection construction, filtering, mapping |
| 3 | `map`, `filter`, `itertools`, `sum`, and similar built-ins | Element transforms and reductions |
| 4 | `for` loop | Side effects, early exit, or cases where alternatives are less clear |

```python
# ✅ 向量化
loss = ((pred - target) ** 2).mean()

# ✅ 推导式
paths = [p for p in root.iterdir() if p.suffix == ".png"]

# ❌ 能向量化却手写循环
total = 0.0
for i in range(len(pred)):
    total += (pred[i] - target[i]) ** 2
```

The rule is not “never use `for`”; it is “do not use `for` when a clearer expression exists.” Readability wins over dogma.

## 10. Pre-commit checklist

- [ ] Every new or modified `.py` file has the section 3 header, and `读取配置` matches actual usage.
- [ ] Implementation files contain no literal configuration; new parameters exist in both `config/default.yaml` and `schema.py`.
- [ ] Runtime validation lives in `checks/<module>_checks.py`, is invoked once at the entry point, and identifies its target. Config validation remains in `schema.py`.
- [ ] Every new module has its own folder and an `__init__.py` with a header and public API re-exports. CLI `run.py` remains at the package root.
- [ ] Source comments are Chinese, explain why, and do not restate the code.
- [ ] Replace loops where vectorization or a comprehension is clearer.
- [ ] `Doc/Index.md` reflects every added or removed file.
