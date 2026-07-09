"""
Label configuration for WESAD dataset analysis.

Provides a centralised mapping from raw WESAD labels (1–4) to target labels
for different analysis modes.  Currently supports:

    binary  (default)  1→0, 2→1, 3→0, 4→0   (Non-Stress / Stress)  — only Stress=1
    3class              1→0, 2→1, 3→2, 4→2   (Baseline / Stress / Amusement·Meditation)

Usage:
    from label_config import LabelConfig
    cfg = LabelConfig('3class')
    target = cfg.LABEL_MAP[raw_label]          # 1 → 0, 2 → 1, 3 → 2, 4 → 2
    name   = cfg.TARGET_NAMES[target]          # 0 → 'Baseline', …
"""


class LabelConfig:
    """Centralised label mapping and metadata for WESAD analysis."""

    # ── Pre-defined modes ──────────────────────────────────────────────────────
    MODES = {
        'binary': {
            'description': 'Binary classification: Non-Stress (0) vs Stress (1)'
                           ' — only Stress=2 is mapped to 1',
            'VALID_LABELS': [1, 2, 3, 4],
            'LABEL_MAP': {1: 0, 2: 1, 3: 0, 4: 0},
            'TARGET_NAMES': {0: 'Non-Stress', 1: 'Stress'},
            'CLASS_NAMES': ['Non-Stress', 'Stress'],
            'N_CLASSES': 2,
            'COLORS': {0: '#2196F3', 1: '#F44336'},
            'label_source': 'wesad_binary 1→0 ; 2→1 ; (3,4)→0',
        },
        '3class': {
            'description': '3-class: Baseline (0), Stress (1), Amusement/Meditation (2)',
            'VALID_LABELS': [1, 2, 3, 4],
            'LABEL_MAP': {1: 0, 2: 1, 3: 2, 4: 2},
            'TARGET_NAMES': {0: 'Baseline', 1: 'Stress', 2: 'Amusement/Meditation'},
            'CLASS_NAMES': ['Baseline', 'Stress', 'Amusement/Meditation'],
            'N_CLASSES': 3,
            'COLORS': {0: '#4CAF50', 1: '#F44336', 2: '#FF9800'},
            'label_source': 'wesad_3class 1→0 ; 2→1 ; (3,4)→2',
        },
    }

    def __init__(self, mode='binary'):
        """
        Parameters
        ----------
        mode : str
            One of 'binary' (default) or '3class'.
        """
        if mode not in self.MODES:
            valid = list(self.MODES.keys())
            raise ValueError(f"Unknown label mode '{mode}'. Valid: {valid}")

        self.mode = mode
        cfg = self.MODES[mode]

        # Raw WESAD labels that are considered valid
        self.VALID_LABELS: list[int] = cfg['VALID_LABELS']

        # Mapping from raw WESAD label → target label
        self.LABEL_MAP: dict[int, int] = cfg['LABEL_MAP']

        # Mapping from target label → human-readable name
        self.TARGET_NAMES: dict[int, str] = cfg['TARGET_NAMES']

        # Ordered list of class names (index = target label)
        self.CLASS_NAMES: list[str] = cfg['CLASS_NAMES']

        # Human-readable description of the label mode
        self.description: str = cfg['description']

        # Number of classes
        self.N_CLASSES: int = cfg['N_CLASSES']

        # Colour per target label (for plotting)
        self.COLORS: dict[int, str] = cfg['COLORS']

        # Human-readable description of the label mapping
        self.label_source: str = cfg['label_source']

    # ── Convenience methods ────────────────────────────────────────────────────

    def map_label(self, raw_label: int) -> int:
        """Map a raw WESAD label to the target label."""
        return self.LABEL_MAP.get(raw_label, raw_label)

    def target_name(self, target_label: int) -> str:
        """Return the human-readable name for a target label."""
        return self.TARGET_NAMES.get(target_label, f'Class {target_label}')

    def __repr__(self) -> str:
        return (f"LabelConfig(mode='{self.mode}', "
                f"{self.N_CLASSES} classes, "
                f"map={self.LABEL_MAP})")