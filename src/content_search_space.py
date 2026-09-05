"""The common predeclared three-parameter content-model search space."""


def sample(trial):
    return dict(lr=trial.suggest_float('lr', 1e-4, 1e-3, log=True),
                dropout=trial.suggest_categorical('dropout', [.1, .2, .3]),
                max_seqlen=trial.suggest_categorical('max_seqlen', [150, 200, 300]))
