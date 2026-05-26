# STEP — Sequential Transmission Evaluation Pipeline

If you use STEP, please cite:
> Lee, H., Çelen, E., Harrison, P. M. C., Anglada-Tort, M., Rijn, P. van,
> Park, M., et al. (2025). *GlobalMood: A cross-cultural benchmark for music
> emotion recognition*. In *Proceedings of the International Society for Music
> Information Retrieval Conference (ISMIR)*.
> https://doi.org/10.48550/arXiv.2505.09539

STEP is a [PsyNet](https://www.psynet.dev/) plugin for running sequential transmission evaluation experiments online. Participants view stimuli (images, audio, video), rate existing tags with stars, flag irrelevant tags, and add new ones. Tags accumulate and converge across participants through an imitation-chain design.

The package ships with:

- **`step`**: the core Python module (trial makers, pages, controls, stimulus classes, i18n support for 36 languages)
- **`demos/STEP-Tag`**: a working PsyNet demo experiment

## Quick Links

- [Source code](https://github.com/pmcharrison/STEP)
- [Issues](https://github.com/pmcharrison/STEP/issues)
- [PsyNet documentation](https://psynetdev.gitlab.io/PsyNet/)

## Installation

### From PyPI

```bash
pip install psynet-step
```

The Python import name is `step`:

```python
from step import StepTag, urls_to_start_nodes
```

### From source (editable)

```bash
git clone https://github.com/pmcharrison/STEP.git
cd STEP
pip install -e .
```

### Optional extras

- `psynet` — pulls in PsyNet and Dallinger (required at runtime; already present in any PsyNet experiment environment)
- `dev` — pytest
- `publish` — build, twine

Install an extra with, e.g.:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from step import StepTag, urls_to_start_nodes
import psynet.experiment
from psynet.timeline import Timeline

class Exp(psynet.experiment.Experiment):
    timeline = Timeline(
        StepTag(
            start_nodes=urls_to_start_nodes(
                [
                    "https://mini-kinetics-psy.s3.amazonaws.com/emotional_prosody/03-01-08-02-02-02-24.wav",
                    "https://s3.amazonaws.com/generalization-datasets/vegetables/images/thaieggplant3.jpg",
                    "https://mini-kinetics-psy.s3.amazonaws.com/mini-kinetics-validation/cut_videos/[zumba]_dLE5YOEqBGs.mp4",
                ]
            ),
            expected_trials_per_participant=2,
            max_iterations=5,
        ),
    )
```

In your experiment's `requirements.txt`:

```
psynet-step==0.1.0
```

## Running the Demo

```bash
cd demos/STEP-Tag
pip install -r constraints.txt
psynet debug local
```

## License

MIT License

