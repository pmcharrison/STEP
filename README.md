## STEP paradigm
### Sequential Transmission Evaluation Pipeline 

### Installation
Install Dallinger and PsyNet as described in the [PsyNet documentation](https://psynetdev.gitlab.io/PsyNet/installation/index.html) and install this package.

### Usage
Once installed you can use it in your experiments by importing the plugins you need:

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


> Don't forget to add the pacakge to your `requirements.txt` file, e.g.:
```bash
step@git+https://github.com/polvanrijn/STEP#egg=step
```
