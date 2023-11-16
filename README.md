## STEP paradigm
### Sequential Transmission Evaluation Pipeline 

### Installation
Install Dallinger and PsyNet as described in the [PsyNet documentation](https://psynetdev.gitlab.io/PsyNet/installation/index.html) and install this package.

### Usage
Once installed you can use it in your experiments by importing the plugins you need:

```python
from step import StepTag
import psynet.experiment
from psynet.timeline import Timeline

class Exp(psynet.experiment.Experiment):
    timeline = Timeline(
        StepTag()
    )
```


> Don't forget to add the pacakge to your `requirements.txt` file, e.g.:
```bash
step@git+https://github.com/polvanrijn/STEP#egg=step
```
