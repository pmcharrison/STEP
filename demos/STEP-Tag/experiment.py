# pylint: disable=unused-import,abstract-method,unused-argument

from psynet.asset import asset
from psynet.consent import NoConsent
from psynet.page import SuccessfulEndPage
from psynet.timeline import Timeline
from psynet.utils import get_logger

import psynet.experiment
from step import (
    StepTag,
)


INITIAL_VOCABULARY = ["cat", "dog", "mouse"]
N_TRIALS_PER_PARTICIPANT = 2


class Exp(psynet.experiment.Experiment):
    label = "STEP Tag demo"

    timeline = Timeline(
        StepTag(
            stimuli={
                "prosody": asset("https://mini-kinetics-psy.s3.amazonaws.com/emotional_prosody/03-01-08-02-02-02-24.wav"),
                "thaieggplant3": asset("https://s3.amazonaws.com/generalization-datasets/vegetables/images/thaieggplant3.jpg"),
                "zumba": asset("https://mini-kinetics-psy.s3.amazonaws.com/mini-kinetics-validation/cut_videos/[zumba]_dLE5YOEqBGs.mp4")
            },
            vocabulary=INITIAL_VOCABULARY,
            expected_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
            max_iterations=5,
        ),
    )

