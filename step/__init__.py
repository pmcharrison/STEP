import html
import os
import hashlib
import random
from builtins import isinstance

from os.path import basename
import string
from typing import List, Optional, Union
from importlib import resources

from dallinger import db
from markupsafe import Markup
from psynet.asset import Asset, ExternalAsset, asset
from psynet.bot import BotResponse
from psynet.data import SQLBase, SQLMixin, register_table
from psynet.experiment import pre_deploy_constant
from psynet.modular_page import (
    AudioPrompt,
    Control,
    ImagePrompt,
    ModularPage,
    VideoPrompt,
)
from psynet.page import InfoPage
from psynet.timeline import join
from psynet.trial.imitation_chain import (
    ImitationChainNetwork,
    ImitationChainNode,
    ImitationChainTrial,
    ImitationChainTrialMaker,
)
from psynet.utils import get_language_dict, get_logger, get_translator
from sqlalchemy import Column, Integer, String

here = os.path.abspath(os.path.dirname(__file__))

__version__ = "0.0.1"

PACKAGE_NAME = "step"

logger = get_logger()


####################
# Helper functions #
####################



def custom_hash(text):
    return hashlib.sha1(text.encode()).hexdigest()


def sanitize_text_for_json(text):
    return text.replace(r"\"", '"').replace("'", "`")


####################
# Defaults         #
####################

DEFAULT_N_STARS = 5
DEFAULT_FLAGGING_THRESHOLD = 2
DEFAULT_FREEZE_ON_N_RATINGS = 3
DEFAULT_FREEZE_ON_MEAN_RATING = 3.0


class StepCandidate:
    def __init__(self, text, hash=None, previous_ratings=None, is_frozen=False, is_flagged=False, is_new=True):
        self.text = text
        if hash is None:
            hash = custom_hash(text)
        self.hash = hash
        if previous_ratings is None:
            previous_ratings = []
        self.previous_ratings = previous_ratings
        self.is_frozen = is_frozen
        self.is_flagged = is_flagged
        self.is_new = is_new


class StepStimulus:
    pass

class StepTagStimulus(StepStimulus):
    def preview(self, assets: dict[str, Asset]):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def prompt(self, text, assets: dict[str, Asset], **kw):
        raise NotImplementedError("This method should be implemented in a subclass.")


class StepTagImage(StepTagStimulus):
    width = 350
    height = 350
    extensions = (".jpg", ".jpeg", ".png", ".gif")

    def preview(self, assets: dict[str, Asset]):
        url = assets["image"].url
        return f'<img src = "{url}" style="max-width:{self.width}px; width:100%; max-height="{self.height}">'

    def prompt(self, text, assets: dict[str, Asset], **kw):
        css = f"""
        <style>
        #prompt-image {{
            max-width: {self.width}px;
            max-height: {self.height}px;
        }}
        </style>
        """
        text = Markup(text + css)
        url = assets["image"].url
        return ImagePrompt(url, text, width=None, height=None)


class StepTagAudioVisual(StepTagStimulus):
    max_height = 350

    def preview(self, assets: dict[str, Asset]):
        image_url = assets["image"].url
        audio_url = assets["audio"].url
        return f"""
        <img src = "{image_url}" style="width:100%; max-height="{self.max_height}">
        <audio controls>
            <source src="{audio_url}">
        </audio>
        """

    def prompt(self, text, assets: dict[str, Asset], **kw):
        css = """
         <style>
            #prompt-text {
                text-align: center;
                font-size: 1.5em;
            }
            #prompt-image, .prompt_img {
                image-rendering: -moz-crisp-edges; /* Firefox */
                image-rendering: -o-crisp-edges; /* Opera */
                image-rendering: -webkit-optimize-contrast; /* Webkit (non-standard naming) */
                image-rendering: crisp-edges;
                -ms-interpolation-mode: nearest-neighbor; /* IE (non-standard property) */
                max-height: 350px;
            }
        </style>
        """
        image_url = assets["image"].url
        audio_url = assets["audio"].url
        prompt = f'<img src = "{image_url}" class="prompt_img"> <h3>{text}</h3>{css}'
        return AudioPrompt(
            audio=audio_url,
            text=Markup(prompt),
        )


class StepTagAudio(StepTagStimulus):
    controls = True
    extensions = (".mp3", ".wav")

    def preview(self, assets: dict[str, Asset]):
        return f'<audio src = "{assets["audio"].url}" controls>'

    def prompt(self, text, assets: dict[str, Asset], **kw):
        return AudioPrompt(assets["audio"].url, text, controls=kw.get("controls", self.controls))


class StepTagVideo(StepTagStimulus):
    controls = False
    width = 300
    extensions = (".mp4", ".webm")

    def preview(self, assets: dict[str, Asset]):
        url = assets["video"].url
        return f'<video src = "{url}" style="max-width:{self.width}px; width:100%;" controls>'

    def prompt(self, text, assets: dict[str, Asset], **kw):
        url = assets["video"].url
        return VideoPrompt(
            url,
            text,
            controls=kw.get("controls", self.controls),
            width=kw.get("width", self.width),
        )


def urls_to_start_nodes(urls):
    return [
        StepNode(
            definition=StepTagDefinition(
                name=url,
                stimulus=url_to_stimulus(url),
            ),
            assets={
                get_url_type(url): asset(url)
            }
        )
        for url in urls
    ]

def get_url_type(url):
    if url.endswith(StepTagAudio.extensions):
        return "audio"
    elif url.endswith(StepTagVideo.extensions):
        return "video"
    elif url.endswith(StepTagImage.extensions):
        return "image"
    else:
        raise ValueError(f"Unsupported file type: {url}")


def url_to_stimulus(url):
    url_type = get_url_type(url)
    if url_type == "audio":
        return StepTagAudio()
    elif url_type == "video":
        return StepTagVideo()
    elif url_type == "image":
        return StepTagImage()


def tags_to_candidates(tags: List[str]):
    assert sorted(tags) == sorted(set(tags)), "Duplicate tags found."
    return [StepCandidate(tag) for tag in tags]


class StepTagControl(Control):
    macro = "step_tag"
    external_template = "step.html"

    def __init__(self, frozen_candidates, unfrozen_candidates, n_stars, translations, auto_complete):
        super().__init__()
        self.frozen_candidates = frozen_candidates
        self.unfrozen_candidates = unfrozen_candidates
        self.n_stars = n_stars
        self.translations = translations
        self.auto_complete = auto_complete

    @property
    def metadata(self):
        return self.__dict__

    def get_bot_response(self, experiment, bot, page, prompt):
        # Get all candidate tags (unfrozen)
        candidate_tags = [c.text for c in self.unfrozen_candidates]

        # Randomly assign a rating (1 to n_stars) or flag (0) to each
        ratings = {}
        for tag in candidate_tags:
            if random.random() < 0.2:  # 20% chance to flag
                ratings[tag] = 0
            else:
                ratings[tag] = random.randint(1, self.n_stars)

        # Generate 1-2 new tags not already present
        existing_tags = set(candidate_tags)
        n_new = random.randint(1, 2)

        new_tags = []
        while len(new_tags) < n_new:
            tag = self.make_random_tag()
            if tag not in existing_tags and tag not in new_tags:
                new_tags.append(tag)

        return BotResponse(
            raw_answer={
                'new_tags': new_tags,
                'ratings': ratings
            }
        )

    def make_random_tag(self):
        return ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))



class StepDefinition:
    def __init__(
            self,
            name: str,
            stimulus: Optional[StepTagStimulus] = None,
            candidates: Optional[List[StepCandidate]] = None
        ):
        if candidates is None:
            candidates = []
        self.name = name
        self.stimulus = stimulus
        self.candidates = candidates


class StepTagDefinition(StepDefinition):
    def __init__(
        self,
        *,
        name: str,
        stimulus: StepTagStimulus,
        tags: Optional[List[str]] = None,
        completed=False,
        candidates: Optional[List[StepCandidate]]=None,
    ):
        if tags is not None and candidates is not None:
            raise ValueError("You cannot both specify tags and candidates.")

        if tags is None:
            tags = []
        if candidates is None:
            candidates = tags_to_candidates(tags)
        self.completed = completed
        super().__init__(name, stimulus, candidates)

    def to_dict(self):
        return {
            "candidates": [candidate.__dict__ for candidate in self.candidates],
            "completed": self.completed,
        }

class StepPage(ModularPage):
    def __init__(
        self,
        flagging_threshold: int = DEFAULT_FLAGGING_THRESHOLD,
        complete_on_n_frozen: int = 1,
        freeze_on_n_ratings: int = DEFAULT_FREEZE_ON_N_RATINGS,
        freeze_on_mean_rating: float = DEFAULT_FREEZE_ON_MEAN_RATING,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.flagging_threshold = flagging_threshold
        self.freeze_on_n_ratings = freeze_on_n_ratings
        self.freeze_on_mean_rating = freeze_on_mean_rating
        self.complete_on_n_frozen = complete_on_n_frozen



class StepTrial(ImitationChainTrial):
    allocated_time = Column(Integer)

    @classmethod
    def _get_trial_time_estimate(cls, trial_maker):
        return trial_maker.mean_time_estimate

    @property
    def visualization_html(self):
        if self.answer is None:
            return "No answer provided yet."

        def print_icon(candidate, cls, style):
            return f"""<label class="btn btn-secondary icon {cls}" style="{style}">
                            <input type="radio" name="{candidate.hash}" id="{candidate.hash}_flag">
                        </label>
                    """

        def print_candidate(candidate, first_candidate=False, n_stars=5):
            if isinstance(candidate, dict):
                candidate = StepCandidate(**candidate)
            rating = candidate.previous_ratings[-1]
            is_flagged = rating == 0
            background = "bg-danger" if is_flagged else "bg-success"
            cls = "mr-1" if first_candidate else "mx-1"
            html_out = f"""
                        <div class="tag-item bg-secondary {cls} my-1">
                            <div class="row tag-name {background}" id="{candidate.hash}-tag">{candidate.text}</div>
                            <div class="btn-group btn-group-toggle" data-toggle="buttons">
                        """
            rating_number = rating if not is_flagged else None
            for n in range(1, n_stars + 1):
                if not is_flagged and n <= rating_number:
                    html_out += print_icon(
                        candidate=candidate,
                        cls=f"star rating{n} selected",
                        style="opacity: 1",
                    )


                else:
                    html_out += print_icon(
                        candidate=candidate,
                        cls=f"star rating{n}",
                        style="opacity: 0.5",
                    )
            if is_flagged and rating == 0:
                html_out += print_icon(
                    candidate=candidate,
                    cls="flag selected",
                    style="opacity: 1",
                )
            else:
                html_out += print_icon(
                    candidate=candidate,
                    cls="flag",
                    style="opacity: 0.5",
                )

            html_out += """</div></div>"""
            return html_out

        new_tags = []
        flagged_tags = []
        completed_tags = []

        tags_html = ""
        if isinstance(self.node.definition, StepTagDefinition):
            definition_candidates = self.node.definition.candidates
        else:
            definition_candidates = self.node.definition["candidates"]
        try:
            candidates = self.answer["candidates"]
        except KeyError:
            logger.warning(
                f"It seems that the answer is not yet provided or malformed for trial {self.id} "
                f"(Participant {self.participant_id})."
            )
            return ""
        for candidate in candidates:
            parent_candidate = [_cand for _cand in definition_candidates if _cand["text"] == candidate["text"]]
            if candidate["is_new"]:
                new_tags.append(candidate["text"])
            else:
                parent_candidate = parent_candidate[0]
                if parent_candidate["is_flagged"] and candidate["is_flagged"]:
                    flagged_tags.append(candidate["text"])
                elif parent_candidate["is_frozen"] and candidate["is_frozen"]:
                    completed_tags.append(candidate["text"])
                else:
                    printed_candidate = print_candidate(candidate, n_stars=self.trial_maker.n_stars)
                    if printed_candidate == "":
                        return ""
                    tags_html += printed_candidate
        out = """
            <style>
            .tag-container {
                display: inline-block;
            }

            .tag-item {
                float: left;
                border-radius: 0.25rem;
                overflow: hidden;
                font-weight: bold;
            }

            .tag-item .btn {
                border: none;
                overflow: hidden;
            }

            .tag-item .btn-group > .btn:not(:first-child) {
                margin: 0;
            }

            .tag-name {
                font-size: 3em;
                color: white;
                padding: 0 0.5em;
            }

            .icon {
                width: 2em;
                height: 2em;
            }

            .icon input {
                display: none;
            }


            .flag {
                background: url('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="white" class="bi bi-flag-fill" viewBox="0 0 16 16"%3E%3Cpath d="M14.778.085A.5.5 0 0 1 15 .5V8a.5.5 0 0 1-.314.464L14.5 8l.186.464-.003.001-.006.003-.023.009a12.435 12.435 0 0 1-.397.15c-.264.095-.631.223-1.047.35-.816.252-1.879.523-2.71.523-.847 0-1.548-.28-2.158-.525l-.028-.01C7.68 8.71 7.14 8.5 6.5 8.5c-.7 0-1.638.23-2.437.477A19.626 19.626 0 0 0 3 9.342V15.5a.5.5 0 0 1-1 0V.5a.5.5 0 0 1 1 0v.282c.226-.079.496-.17.79-.26C4.606.272 5.67 0 6.5 0c.84 0 1.524.277 2.121.519l.043.018C9.286.788 9.828 1 10.5 1c.7 0 1.638-.23 2.437-.477a19.587 19.587 0 0 0 1.349-.476l.019-.007.004-.002h.001"/%3E%3C/svg%3E') no-repeat;
                background-position: center;
            }

            .star {
                background-image: url('data:image/svg+xml,%3C%3Fxml version="1.0" encoding="utf-8"%3F%3E%3Csvg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 16 16" style="enable-background:new 0 0 16 16;" xml:space="preserve"%3E%3Cstyle type="text/css"%3E.st0%7Bfill:%23FFFFFF;%7D%3C/style%3E%3Cg%3E%3Cpolygon class="st0" points="8,2.6 9.8,6.1 13.7,6.7 10.8,9.5 11.5,13.4 8,11.5 4.5,13.4 5.2,9.5 2.3,6.7 6.2,6.1 "/%3E%3C/g%3E%3C/svg%3E');
            }

            path {
                fill: #fff;
            }

            .badge {
                font-size: 1em;
            }

            .media-item {
                float: left;
                width: 100%;
            }

            .icon.selected {
                color: var(--bs-btn-active-color) !important;
                background-color: var(--bs-btn-active-bg) !important;
                border-color: var(--bs-btn-active-border-color) !important;
            }
            </style>
            """


        def print_tags(tags, tag_type):
            if tag_type == "new":
                tag_cls = 'text-bg-dark'
            elif tag_type == "flagged":
                tag_cls = 'text-bg-danger'
            elif tag_type == "completed":
                tag_cls = 'text-bg-success'
            else:
                raise ValueError(f"Unknown tag type: {tag_type}")
            if len(tags) == 0:
                return ""
            tags = [f"<span class='badge rounded-pill {tag_cls}'>{tag}</span>" for tag in tags]
            return f"<span>{' '.join(tags)}</span><br>"

        if len(flagged_tags) > 0:
            out += print_tags(flagged_tags, "flagged")
        if len(completed_tags) > 0:
            out += print_tags(completed_tags, "completed")
        if len(new_tags) > 0:
            out += print_tags(new_tags, "new")

        out += tags_html

        return out



class StepTagPage(StepPage):
    def __init__(
        self,
        *,
        stimulus: StepTagStimulus,
        frozen_candidates: List[StepCandidate],
        unfrozen_candidates: List[StepCandidate],
        available_tags: List[str],
        used_tags: List[str],
        n_stars: int,
        javascript_translations: dict,
        jinja_translations: dict,
        assets: dict[str, Asset],
        flagging_threshold: int = DEFAULT_FLAGGING_THRESHOLD,
        complete_on_n_frozen: int = 2,
        auto_complete: bool = True,
        freeze_on_n_ratings: int = DEFAULT_FREEZE_ON_N_RATINGS,
        freeze_on_mean_rating: float = DEFAULT_FREEZE_ON_MEAN_RATING,
        **kwargs,
    ):
        self.stimulus = stimulus
        self.frozen_candidates = frozen_candidates
        self.unfrozen_candidates = unfrozen_candidates
        self.used_tags = used_tags
        escaped_js_translations = {'translations': {
            k: html.escape(v) for k, v in javascript_translations['translations'].items()
        }
        }

        super().__init__(
            flagging_threshold,
            complete_on_n_frozen,
            freeze_on_n_ratings,
            freeze_on_mean_rating,
            label="StepTagTrial",
            prompt=stimulus.prompt(text="", assets=assets),
            control=StepTagControl(
                frozen_candidates=frozen_candidates,
                unfrozen_candidates=unfrozen_candidates,
                n_stars=n_stars,
                translations=jinja_translations,
                auto_complete=auto_complete,
            ),
            js_vars={
                **escaped_js_translations,
                "available_tags": [html.escape(tag) for tag in available_tags],
                "used_tags": [html.escape(tag) for tag in used_tags],
            },
            **kwargs,
        )


class StepTagTrial(StepTrial):
    time_estimate = 0  # Default Value

    def show_trial(self, experiment, participant):
        available_tags = self.var.get("available_tags")
        used_tags = self.var.get("used_tags")
        frozen_candidates = self.var.get("frozen_candidates")
        unfrozen_candidates = self.var.get("unfrozen_candidates")

        stimulus = self.node.definition.stimulus

        return StepTagPage(
            stimulus=stimulus,
            assets=self.assets,
            frozen_candidates=frozen_candidates,
            unfrozen_candidates=unfrozen_candidates,
            available_tags=available_tags,
            used_tags=used_tags,
            n_stars=self.trial_maker.n_stars,
            complete_on_n_frozen=self.trial_maker.complete_on_n_frozen,
            auto_complete=self.trial_maker.auto_complete,
            flagging_threshold=self.trial_maker.flagging_threshold,
            javascript_translations=self.trial_maker.get_javascript_translations(),
            jinja_translations=self.trial_maker.get_jinja_translations(),
        )


class StepNetwork(ImitationChainNetwork):
    pass


class StepNode(ImitationChainNode):
    time_estimate = Column(Integer)
    stimulus_name = Column(String)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.degree > 0:
            self.assets = self.parent.assets
        self.stimulus_name = self.definition.name

    def create_initial_seed(self, experiment, participant):
        return {}

    def create_definition_from_seed(self, seed, experiment, participant):
        return seed

    def cast_definition(self, definition):
        assert isinstance(definition, StepTagDefinition)
        # if isinstance(definition, dict):
        #     stimulus = definition["stimulus"]
        #     candidates = [StepCandidate(**cand) for cand in definition['candidates']]
        #     return StepTagDefinition(
        #         name=definition["name"],
        #         stimulus=stimulus,
        #         candidates=candidates,
        #         completed=definition["completed"]
        #     )
        return definition

    def get_definitions(self):
        return [self.cast_definition(self.definition)]

    def estimate_time(self, rating_time_estimate, creating_time_estimate, view_time_estimate):
        if self.time_estimate is None:
            # median ratings + creations in pilot
            self.time_estimate = 4 * rating_time_estimate + creating_time_estimate
        if self.time_estimate < view_time_estimate:
            self.time_estimate = view_time_estimate
        return self.time_estimate

    def summarize_trials(self, trials: list, experiment, participant):
        """
        Bugfix such that we always return the first answer, also if multiple trials were made. This can happen in rare
        cases in which participants mistakenly are assigned to the same trial at the same time.
        """

        return trials[0].answer


def freeze_candidate_if_needed(candidate, freeze_on_n_ratings, freeze_on_mean_rating):
    if len(candidate.previous_ratings) >= freeze_on_n_ratings:
        average_rates = sum(candidate.previous_ratings) / len(
            candidate.previous_ratings
        )
        candidate.is_frozen = average_rates >= freeze_on_mean_rating
    return candidate


class StepTrialMaker(ImitationChainTrialMaker):
    """
    This base class for STEP (Sequential Transmission Evaluation Pipeline). STEP is an evaluation pipeline that allows
    users to rate or flag existing annotations or provide new ones.

    Each annotation carries a list of previous ratings (at start it's empty). If a candidate is flagged, it no longer
    receives ratings and can be revived if it is added again.

    If certain annotations receive a minimal number of ratings (`freeze_on_n_ratings`) with a mean rating above a
    certain threshold (`freeze_on_mean_rating`), they are considered frozen and no longer receive ratings. A chain can
    stop early if a certain number of annotations are frozen (`complete_on_n_frozen`).



    Parameters
    ----------

    expected_trials_per_participant: int
        The number of trials per participant

    max_iterations: int
        The maximum number of iterations per trial

    flagging_threshold: int
        The number of times a candidate must be flagged before it is dropped. The default is 2.

    n_stars: int
        The number of stars to use for rating. The default is 5.

    freeze_on_n_ratings: int
        The minimum number of ratings an annotation must receive before it is frozen. The default is 3.

    freeze_on_mean_rating: float
        The minimum mean rating an annotation must receive before it is frozen. The default is 3.0.

    complete_on_n_frozen: int
        The number of frozen annotations required to end the chain early. The default is 1.

    rating_time_estimate: int
        The number of seconds estimated for rating. The default is 1.

    creating_time_estimate: int
        The number of seconds estimated for creating. The default is 4.

    node_class: class
        The class to use for nodes. The default is StepNode.

    """

    time_estimate_per_trial = 0

    def __init__(
        self,
        expected_trials_per_participant,
        max_iterations,
        label="StepTrialMaker",
        max_trials_per_participant: Union[None, int, str] = None,
        flagging_threshold: int = 2,
        n_stars: int = DEFAULT_N_STARS,
        freeze_on_n_ratings: int = 3,
        freeze_on_mean_rating: float = 3.0,
        complete_on_n_frozen: int = 1,
        rating_time_estimate: int = 1,
        creating_time_estimate: int = 4,
        view_time_estimate: int = 1,
        node_class=StepNode,
        debug=False,
        auto_complete=True,
        show_instructions=True,
        practice_stimuli: Optional[List[StepStimulus]] = None,
        *args,
        **kwargs,
    ):
        if practice_stimuli is not None:
            raise NotImplementedError("Practice stimuli are not supported yet.")

        kwargs["id_"] = label
        kwargs["network_class"] = StepNetwork
        kwargs["node_class"] = node_class
        kwargs["chain_type"] = "across"
        kwargs["max_nodes_per_chain"] = max_iterations
        kwargs["expected_trials_per_participant"] = expected_trials_per_participant
        if max_trials_per_participant is not None:
            kwargs["max_trials_per_participant"] = max_trials_per_participant
        kwargs["recruit_mode"] = "n_trials"
        assert kwargs["start_nodes"] is not None, "You must specify start nodes."

        if isinstance(kwargs["start_nodes"], list):
            assert (
                len(kwargs["start_nodes"]) >= expected_trials_per_participant
            ), "You must specify at least as many start nodes as trials per participant."
            chains_per_experiment = len(kwargs["start_nodes"])
        else:
            chains_per_experiment = None

        kwargs["chains_per_experiment"] = chains_per_experiment

        assert (
            freeze_on_mean_rating <= n_stars
        ), "The freeze_on_mean_rating must be smaller than or equal to n_stars."
        assert freeze_on_n_ratings >= 1, "The freeze_on_n_ratings must be at least 1."
        assert (
            freeze_on_n_ratings <= max_iterations
        ), "The freeze_on_n_ratings must be smaller than or equal to max_iterations."
        assert freeze_on_n_ratings >= 1, "The freeze_on_n_ratings must be at least 1."

        self.expected_trials_per_participant = expected_trials_per_participant
        self.view_time_estimate = view_time_estimate

        # This logic no longer works because the nodes might not be available yet (they could just be a callable)
        # time_estimates = [
        #     node.estimate_time(rating_time_estimate, creating_time_estimate, view_time_estimate)
        #     for node in kwargs["start_nodes"]
        # ]
        # self.mean_time_estimate = sum(time_estimates) / len(time_estimates)

        self.mean_time_estimate = 4 * rating_time_estimate + creating_time_estimate

        self.time_budget = (
            self.mean_time_estimate * self.expected_trials_per_participant
        )

        self.debug = debug
        self.auto_complete = auto_complete
        self.show_instructions = show_instructions
        if practice_stimuli is None:
            practice_stimuli = []
        self.practice_stimuli = practice_stimuli
        self.n_practice_trials = len(practice_stimuli)

        super().__init__(*args, **kwargs)
        self.flagging_threshold = flagging_threshold
        self.n_stars = n_stars
        self.freeze_on_n_ratings = freeze_on_n_ratings
        self.freeze_on_mean_rating = freeze_on_mean_rating
        self.complete_on_n_frozen = complete_on_n_frozen
        self.rating_time_estimate = rating_time_estimate
        self.creating_time_estimate = creating_time_estimate

    def custom_network_filter(self, candidates, participant):
        visited_trials = self.trial_class.query.filter_by(
            trial_maker_id=self.id, participant_id=participant.id
        ).all()
        total_allocated_time = sum([trial.allocated_time for trial in visited_trials])
        time_left = self.time_budget - total_allocated_time
        logger.info(f"Participant {participant.id} has {time_left} seconds left.")
        candidates = [
            candidate
            for candidate in candidates
            if candidate.head is not None and candidate.head.estimate_time(
                self.rating_time_estimate, self.creating_time_estimate, self.view_time_estimate
            )
               <= time_left
        ]
        return candidates

    def _update_candidate(self, candidate, rating):
        candidate.previous_ratings.append(rating)
        candidate = freeze_candidate_if_needed(
            candidate, self.freeze_on_n_ratings, self.freeze_on_mean_rating
        )
        n_total_flags = sum([rating == 0 for rating in candidate.previous_ratings])
        has_to_be_flagged = n_total_flags > 0 and n_total_flags % self.flagging_threshold == 0
        candidate.is_flagged = has_to_be_flagged
        return candidate

    def update_candidate(
        self,
        candidate,
        parsed_candidate_text,
        provided_n_frozen,
        new_content_container,
        rating,
    ):
        n_frozen = provided_n_frozen
        if candidate.is_frozen:
            n_frozen += 1
        elif candidate.is_flagged:
            if parsed_candidate_text in new_content_container:
                candidate.previous_ratings.append(0)
                candidate.is_flagged = False
                new_content_container.pop(parsed_candidate_text)
        else:
            candidate = self._update_candidate(candidate, rating)
            if candidate.is_frozen:
                n_frozen += 1
        return new_content_container, candidate, n_frozen

    def decide_if_network_is_full(self, trial):
        node = trial.origin
        node_definitions = node.get_definitions()
        node_candidates = sum(
            [
                definition.candidates
                for definition in node_definitions
                if not definition.completed
            ],
            [],
        )
        node_candidates_num_frozen = sum(
            [candidate.is_frozen for candidate in node_candidates]
        )
        if node_candidates_num_frozen >= self.complete_on_n_frozen:
            node.network.full = True
            # db.session.commit()

    def format_answer(self, trial, raw_answer):
        raise NotImplementedError("Must be implemented by subclass.")

    def finalize_trial(self, answer, trial, experiment, participant):
        answer = self.format_answer(trial, answer)
        trial.answer = answer
        super().finalize_trial(answer, trial, experiment, participant)
        self.decide_if_network_is_full(trial)

    @classmethod
    def get_instructions_before_practice(cls, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")

    def get_practice(self):
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    def get_instructions_after_practice(cls, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    def get_instructions(cls, debug=False, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")

    @property
    def introduction(self):
        pages = []

        if self.show_instructions:
            pages.append(self.get_instructions())

        if self.n_practice_trials > 0:
            pages.append(self.get_instructions_before_practice())
            pages.append(self.get_practice())
            pages.append(self.get_instructions_after_practice())

        return join(*pages) if len(pages) > 0 else None



@register_table
class Vocabulary(SQLBase, SQLMixin):
    __tablename__ = "vocabulary"

    word = Column(String(255), nullable=False)

    @classmethod
    def extend(cls, new_words: List[str]):
        for word in new_words:
            if Vocabulary.query.filter_by(word=word).count() == 0:
                new_word = Vocabulary(word=word)
                db.session.add(new_word)
        # db.session.commit()


class StepTag(StepTrialMaker):
    response_timeout_sec = 5 * 60  # 5 minutes

    def __init__(
        self,
        stimuli: Union[dict[str, Asset], callable],
        vocabulary: List[str] = None,
        complete_on_n_frozen: int = 2,
        rating_time_estimate: int = 3,  # set based on pilot data
        creating_time_estimate: int = 6,  # set based on pilot data
        # deposit_assets: bool = True,
        *args,
        **kwargs,
    ):
        assert "start_nodes" not in kwargs

        self.n_stimuli = pre_deploy_constant(
            "n_stimuli",
            lambda: len(stimuli()) if callable(stimuli) else len(stimuli),
        )

        if complete_on_n_frozen == "n_stimuli":
            complete_on_n_frozen = self.n_stimuli

        for key in [
            "expected_trials_per_participant",
            "max_trials_per_participant",
            "max_iterations",
            "complete_on_n_frozen",
        ]:
            if key in kwargs and kwargs[key] == "n_stimuli":
                kwargs[key] = self.n_stimuli

        def start_nodes():
            nonlocal stimuli

            if callable(stimuli):
                stimuli = stimuli()

            assert isinstance(stimuli, dict), f"stimuli must be a dict, not {type(stimuli)}"

            return [
                StepNode(
                    definition=StepTagDefinition(
                        name=name,
                        stimulus=url_to_stimulus(asset.extension),
                    ),
                    assets={
                        get_url_type(asset.extension): asset
                    }
                )
                for name, asset in stimuli.items()
            ]

        kwargs["start_nodes"] = start_nodes

        if vocabulary is None:
            vocabulary = []
        self.vocabulary = vocabulary


        # Register assets
        assets = []
        # if deposit_assets:
        #     for node in kwargs["start_nodes"]:
        #         if isinstance(node.definition, StepTagDefinition):
        #             stimulus = node.definition.stimulus
        #             url = stimulus.url
        #         else:
        #             url = node.definition['stimulus']
        #         fname = basename(url)
        #         short_hash = hashlib.sha1(fname.encode()).hexdigest()
        #         # same filename is okay if they are from a different url
        #         local_key = f"{short_hash}_{fname}"
        #         asset = ExternalAsset(url=url, local_key=local_key)
        #         node.asset = asset
        #         assets.append(asset)

        if "trial_class" in kwargs:
            trial_class=kwargs["trial_class"]
            del kwargs["trial_class"]
        else:
            trial_class = StepTagTrial

        super().__init__(
            trial_class=trial_class,
            complete_on_n_frozen=complete_on_n_frozen,
            rating_time_estimate=rating_time_estimate,
            creating_time_estimate=creating_time_estimate,
            assets=assets,
            *args,
            **kwargs,
        )

    # Implement this in your subclass
    # @expose_to_api("validate_tag")
    # @staticmethod
    # def validate_tag(tag):
    #     return {
    #         "success": True,
    #     }

    def format_answer(self, trial, raw_answer):
        used_tags = trial.var.get("used_tags")
        frozen_candidates = trial.var.get("frozen_candidates")
        unfrozen_candidates = trial.var.get("unfrozen_candidates")
        hidden_candidates = trial.var.get("hidden_candidates")
        candidates = unfrozen_candidates + frozen_candidates + hidden_candidates
        all_tags = [candidate.text for candidate in candidates]
        shown_tags = [
            candidate.text
            for candidate in candidates
            if not (candidate.is_frozen or candidate.is_flagged)
        ]
        rated_tags = sorted(raw_answer["ratings"].keys())
        assert sorted(shown_tags) == rated_tags, "Tags must all be rated."
        _new_tags = [html.unescape(tag) for tag in raw_answer["new_tags"]]
        new_tags = [tag for tag in _new_tags if tag not in all_tags]


        trial.allocated_time = (
            len(rated_tags) * self.rating_time_estimate
            + len(new_tags) * self.creating_time_estimate
        )

        # Update existing candidates
        n_frozen = 0
        new_tags_dict = {t: None for t in new_tags}

        for i in range(len(unfrozen_candidates)):
            candidate = unfrozen_candidates[i]
            parsed_candidate_text = sanitize_text_for_json(html.unescape(candidate.text))
            candidate_rating = raw_answer["ratings"][candidate.text]
            new_tags_dict, candidate, n_frozen = self.update_candidate(
                candidate=candidate,
                parsed_candidate_text=parsed_candidate_text,
                provided_n_frozen=n_frozen,
                new_content_container=new_tags_dict,
                rating=candidate_rating,
            )
            unfrozen_candidates[i] = candidate

        # add new words to vocabulary
        new_tags = list(new_tags_dict.keys())
        Vocabulary.extend(new_tags)

        # add new candidates to trial
        for tag in new_tags:
            candidates.append(StepCandidate(text=tag))

        for i, candidate in enumerate(candidates):
            if candidate.text in _new_tags:
                candidates[i].is_flagged = False
                candidates[i].is_new = True
            else:
                candidates[i].is_new = False

        assert isinstance(trial.node.definition, StepTagDefinition)
        name = trial.node.definition.name
        stimulus = trial.node.definition.stimulus

        return StepTagDefinition(
            name=name,
            stimulus=stimulus,
            candidates=candidates,
            completed=n_frozen >= self.complete_on_n_frozen,
        )

    def prepare_trial(self, experiment, participant):
        trial, trial_status = super().prepare_trial(experiment, participant)
        if trial_status in ["wait", "exit"]:
            return trial, trial_status

        if isinstance(trial.definition, StepTagDefinition):
            candidates = trial.definition.candidates
        else:
            candidates = [StepCandidate(**cand) for cand in trial.definition['candidates']]

        candidates = random.sample(candidates, len(candidates))  # shuffle the tags
        used_candidates = [
            candidate
            for candidate in candidates
            if not candidate.is_flagged
        ]

        hidden_candidates = [
            candidate
            for candidate in candidates
            if candidate.is_flagged
        ]
        used_contents = [candidate.text for candidate in used_candidates]
        logger.info(f"Used tags: {used_contents} (Trial {trial.id}, Participant {participant.id})")
        logger.info(
            f"Hidden tags: {[candidate.text for candidate in hidden_candidates]} (Trial {trial.id}, Participant {participant.id})")
        frozen_candidates = [
            candidate for candidate in used_candidates if candidate.is_frozen
        ]
        unfrozen_candidates = [
            candidate for candidate in used_candidates if not candidate.is_frozen
        ]
        available_tags = [
            vocabulary_item.word for vocabulary_item in Vocabulary.query.all()
        ]
        trial.var.set("available_tags", available_tags)
        trial.var.set("used_tags", used_contents)
        trial.var.set("frozen_candidates", frozen_candidates)
        trial.var.set("unfrozen_candidates", unfrozen_candidates)
        trial.var.set("hidden_candidates", hidden_candidates)

        return trial, trial_status

    def create_networks_across(self, experiment):
        super().create_networks_across(experiment)

        # TODO: It would be more efficient if we could get the nodes directly from create_networks_across
        start_nodes = db.session.query(StepNode).filter_by(trial_maker_id=self.id).all()

        initial_tags = [
            candidate.text
            for node in start_nodes
            for candidate in node.definition.candidates
        ]

        initial_vocabulary = list(set(initial_tags + self.vocabulary))
        Vocabulary.extend(initial_vocabulary)

    @staticmethod
    def get_instructions_without_tags():
        _p = get_translator(context=True)
        out = '<h3 for="new_tags">' + _p("STEP-Tag", "Add some initial tags") + "</h3>"
        out += '<div class="alert alert-primary" role="alert">'
        out += " ".join(
            [
                _p("STEP-Tag", "Type in tags describing the stimulus."),
                _p(
                    "STEP-Tag",
                    "You can either select tags from a dropdown list or create entirely new ones.",
                ),
                _p(
                    "STEP-Tag",
                    "Submit your response for a new tag by pressing the {ENTER_KEY} key.",
                ).format(ENTER_KEY="<kbd>enter</kbd>"),
                "<strong>"
                + _p("STEP-Tag", "You can add more than one tag.")
                + "</strong>",
            ]
        )
        out += "</div>"
        return out

    @staticmethod
    def get_instructions_with_tags():
        _p = get_translator(context=True)
        out = '<h3 for="new_tags">' + _p("STEP-Tag", "Are any tags missing?") + "</h3>"
        out += '<div class="alert alert-primary" role="alert">'
        out += " ".join(
            [
                _p(
                    "STEP-Tag",
                    "Type in tags describing the stimulus, that are missing above.",
                ),
                _p(
                    "STEP-Tag",
                    "You can either select tags from a dropdown list or create entirely new ones.",
                ),
                _p(
                    "STEP-Tag",
                    "Submit your response for a new tag by pressing the {ENTER_KEY} key.",
                ).format(ENTER_KEY="<kbd>enter</kbd>"),
                "<strong>"
                + _p("STEP-Tag", "You can add more than one tag.")
                + "</strong>",
            ]
        )
        out += "</div>"
        return out

    @classmethod
    def get_jinja_translations(cls):
        _ = get_translator()
        _p = get_translator(context=True)
        return {
            "title_frozen": _p("STEP-Tag", "The following tags are already completed:"),
            "title_unfrozen": _p("STEP-Tag", "Mark the existing tags"),
            "type_more": _p("STEP-Tag", "Type more tags"),
            'title_tags': _p("STEP-Tag", "Add tags"),
            "next": _("Next"),
            "instructions_without_tags": cls.get_instructions_without_tags(),
            "instructions_with_tags": cls.get_instructions_with_tags(),
        }

    @staticmethod
    def get_javascript_translations():
        _p = get_translator(context=True)
        return {
            "translations": {
                "rate_all_tags": _p("STEP-Tag", "You need to rate all tags!"),
                "specify_one_tag": _p(
                    "STEP-Tag", "You need to supply at least one new tag!"
                ),
                "cannot_submit": _p(
                    "STEP-Tag", "There is still some unsubmitted text in the tag field."
                )
                                 + " "
                                 + _p("STEP-Tag", "Please submit or delete it first."),
                "whitespaces": " ".join(
                    [
                        _p(
                            "STEP-Tag", "Your tag contains {N_WHITESPACES} whitespaces."
                        ),
                        _p("STEP-Tag", "Is it necessary to use this many whitespaces?"),
                        _p("STEP-Tag", "If you are sure, you can go ahead."),
                    ]
                ),
            }
        }

    @classmethod
    def get_instructions_before_practice(cls):
        _p = get_translator(context=True)
        return InfoPage(
            Markup(
                " ".join(
                    [
                        _p("STEP-Tag", "Let's start with a practice round!"),
                    ]
                )
            ),
            time_estimate=2,
        )

    def get_practice(self):
        practice_pages = []
        for stimulus in self.practice_stimuli:
            practice_pages.append(
                StepTagPractice(stimulus=stimulus, time_estimate=10)
            )
        return join(*practice_pages)

    @classmethod
    def get_instructions_after_practice(cls, **kwargs):
        _p = get_translator(context=True)
        return InfoPage(
            Markup(
                _p("STEP-Tag", "That was a practice round.")
                + " "
                + _p("STEP-Tag", "Now, let's start the main experiment!")
            ),
            time_estimate=3,
        )

    @classmethod
    def get_first_instructions(cls):
        _p = get_translator(context=True)
        return InfoPage(
            Markup(
                " ".join(
                    [
                        _p("STEP-Tag", "Thanks for participating in this experiment!"),
                        "<br><br>",
                        _p("STEP-Tag", "In this experiment you will:"),
                        "<ul>",
                        "<li>"
                        + _p("STEP-Tag", "View a stimulus and describe it with tags.")
                        + "</li>",
                        "<li>"
                        + _p("STEP-Tag", "Rate tags from other participants")
                        + "</li>",
                        "<li>"
                        + _p("STEP-Tag", "Add new tags that you think are missing")
                        + "</li>",
                        "</ul>",
                    ]
                )
            ),
            time_estimate=5,
        )

    @classmethod
    def get_rating_instructions(cls):
        _p = get_translator(context=True)

        return InfoPage(
            Markup(
                " ".join(
                    [
                        "<h3>" + _p("STEP-Tag", "Rating") + "</h3>",
                        "<ul>",
                        "<li>"
                        + _p(
                            "STEP-Tag",
                            "After viewing the stimulus, you will see tags from other participants that describe the stimulus.",
                        )
                        + "</li>",
                        "<li>"
                        + _p(
                            "STEP-Tag",
                            "You should rate the relevance of each tag by clicking the appropriate amount of stars.",
                        ),
                        _p(
                            "STEP-Tag",
                            "1 star means not very relevant, 5 stars means very relevant.",
                        )
                        + "</li>",
                        "<li>"
                        + _p(
                            "STEP-Tag",
                            "If you think that the tag is a mistake or completely irrelevant, you should flag it by clicking the flag icon.",
                        )
                        + "</li>",
                        "<li>"
                        + _p(
                            "STEP-Tag",
                            "If you are the first person seeing the stimulus, you may see no previous tags.",
                        )
                        + "</li>",
                        "</ul>",
                    ]
                )
            ),
            time_estimate=5,
        )

    @classmethod
    def get_creating_instructions(cls):
        _p = get_translator(context=True)

        return InfoPage(
            Markup(
                " ".join(
                    [
                        "<h3>" + _p("STEP-Tag", "Creating") + "</h3>",
                        "<ul>",
                        "<li>"
                        + _p(
                            "STEP-Tag",
                            "After viewing the stimulus, you can add your own tags that describe the stimulus.",
                        )
                        + "</li>",
                        "<li>"
                        + _p(
                            "STEP-Tag",
                            "Your tag will then be rated by other players who are playing the game simultaneously.",
                        )
                        + "</li>",
                        "</ul>",
                        "<div class='alert alert-primary' role='alert'>",
                        "<strong>"
                        + _p("STEP-Tag", "Keep tags short:")
                        + "</strong> "
                        + _p(
                            "STEP-Tag",
                            "A word like 'green grass' should rather be submitted as 'green' and 'grass', whereas a compound word such as 'red wine' cannot be separated, since 'red wine' means something different than just 'red' and 'wine'.",
                        ),
                        "</div>",
                    ]
                )
            ),
            time_estimate=5,
        )

    @classmethod
    def get_instructions(cls, debug=False, **kwargs):
        _p = get_translator(context=True)

        return join(
            cls.get_first_instructions(),
            cls.get_rating_instructions(),
            cls.get_creating_instructions(),
        )

    @classmethod
    def extra_files(cls):
        return [
            *super().extra_files(),
            (
                resources.files(PACKAGE_NAME) / "static/",
                f"/static/{PACKAGE_NAME}/",
            ),
            (
                resources.files(PACKAGE_NAME) / "templates/step.html",
                "/templates/step.html/",
            ),
        ]


class StepTagPractice(StepTagPage):
    def __init__(
        self,
        stimulus: StepStimulus,
        time_estimate: float,
        frozen_candidates: List[StepCandidate] = [],
        unfrozen_candidates: List[StepCandidate] = [],
        available_tags: List[str] = [],
        used_tags: List[str] = [],
        n_stars: int = DEFAULT_N_STARS,
        flagging_threshold: int = DEFAULT_FLAGGING_THRESHOLD,
        freeze_on_n_ratings: int = DEFAULT_FREEZE_ON_N_RATINGS,
        freeze_on_mean_rating: float = DEFAULT_FREEZE_ON_MEAN_RATING,
        step_class=StepTag,
    ):
        self.stimulus = stimulus
        self.frozen_candidates = frozen_candidates
        self.unfrozen_candidates = unfrozen_candidates
        self.flagging_threshold = flagging_threshold
        self.used_tags = used_tags
        self.freeze_on_n_ratings = freeze_on_n_ratings
        self.freeze_on_mean_rating = freeze_on_mean_rating
        jinja_translations = step_class.get_jinja_translations()
        javascript_translations = step_class.get_javascript_translations()

        super().__init__(
            stimulus=stimulus,
            frozen_candidates=frozen_candidates,
            unfrozen_candidates=unfrozen_candidates,
            available_tags=available_tags,
            used_tags=used_tags,
            n_stars=n_stars,
            flagging_threshold=flagging_threshold,
            javascript_translations=javascript_translations,
            jinja_translations=jinja_translations,
            freeze_on_n_ratings=freeze_on_n_ratings,
            freeze_on_mean_rating=freeze_on_mean_rating,
            time_estimate=time_estimate,
        )
