"""Preserve old navigation and annotation rows when their vocabularies grow."""


_EXPANDABLE = (
    "visual.navigation_token.weight",
    "trajectory_decoder.navigation.weight",
    "visual.annotation_state_head.weight",
    "visual.annotation_state_head.bias",
    "visual.annotation_reason_head.weight",
    "visual.annotation_reason_head.bias",
)


def compatible_driving_state(current, saved):
    """Return compatible tensors and names safe for optimizer-state reuse.

    Expanded rows retain old weights; new rows keep the model's initialization.
    Optimizer moments for expanded tensors must be initialized afresh.
    """
    compatible = {}
    exact_names = set()
    for name, value in saved.items():
        if name not in current:
            continue
        target = current[name]
        if tuple(value.shape) == tuple(target.shape):
            compatible[name] = value
            exact_names.add(name)
        elif (name in ("visual.navigation_token.weight", "trajectory_decoder.navigation.weight")
              and value.shape[0] == 7 and target.shape[0] == 5
              and tuple(value.shape[1:]) == tuple(target.shape[1:])):
            # Interim seven-command models: retain the original route rows.
            compatible[name] = value[:5].clone()
        elif (name in _EXPANDABLE and value.ndim == target.ndim and
              value.shape[0] < target.shape[0] and
              tuple(value.shape[1:]) == tuple(target.shape[1:])):
            expanded = target.clone()
            expanded[:value.shape[0]] = value.to(expanded)
            compatible[name] = expanded
    return compatible, exact_names
