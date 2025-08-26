# pylint: disable=unused-import,abstract-method,unused-argument

from psynet.asset import asset
from psynet.consent import NoConsent
from psynet.page import SuccessfulEndPage
from psynet.timeline import Timeline
from psynet.utils import get_logger

import psynet.experiment
from step import (
    StepCandidate,
    StepNode,
    StepTag,
    StepTagDefinition,
    StepTagImage,
    url_to_stimulus,
    urls_to_start_nodes,
)

logger = get_logger()

LABEL = "STEP Tag demo"
INITIAL_VOCABULARY = ["cat", "dog", "mouse"]
N_TRIALS_PER_PARTICIPANT = 2

# Not supported for now
# # You can also define your own start nodes with pre-existing tags
# def get_custom_start_nodes():
#     # You can also customize your own start nodes
#     return [
#         StepNode(
#             definition=StepTagDefinition(
#                 name="thaieggplant3",
#                 stimulus=StepTagImage(),
#                 candidates=[
#                     StepCandidate("example_frozen", is_frozen=True),
#                     StepCandidate("example_unfrozen", is_frozen=False),
#                     StepCandidate("example_frozen2", is_frozen=True),
#                     StepCandidate("example_unfrozen2", is_frozen=False),
#                 ],
#             ),
#             assets={
#                 "image": asset(
#                     "https://s3.amazonaws.com/generalization-datasets/vegetables/images/thaieggplant3.jpg"
#                 ),
#             },
#         )
#     ]


class Exp(psynet.experiment.Experiment):
    label = LABEL

    config = {
        "initial_recruitment_size": 1,
        "title": LABEL,
        "description": "This is a demo of the STEP paradigm.",
    }

    timeline = Timeline(
        NoConsent(),
        StepTag(
            stimuli={
                "prosody": asset("https://mini-kinetics-psy.s3.amazonaws.com/emotional_prosody/03-01-08-02-02-02-24.wav"),
                "thaieggplant3": asset("https://s3.amazonaws.com/generalization-datasets/vegetables/images/thaieggplant3.jpg"),
                "zumba": asset("https://mini-kinetics-psy.s3.amazonaws.com/mini-kinetics-validation/cut_videos/[zumba]_dLE5YOEqBGs.mp4")
            },
            vocabulary=INITIAL_VOCABULARY,
            expected_trials_per_participant=N_TRIALS_PER_PARTICIPANT,
            max_iterations=5,
            # practice_stimuli=[
            #     asset("https://s3.amazonaws.com/generalization-datasets/vegetables/images/thaieggplant3.jpg")
            # ],
        ),
        SuccessfulEndPage(),
    )
