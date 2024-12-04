import html
import os
import hashlib
import random

from os.path import basename
from typing import List
from importlib import resources

from dallinger import db
from markupsafe import Markup
from psynet.asset import ExternalAsset
from psynet.data import SQLBase, SQLMixin, register_table
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
from psynet.utils import get_language_dict, get_logger
from sqlalchemy import Column, Integer, String

here = os.path.abspath(os.path.dirname(__file__))

__version__ = "0.0.1"

PACKAGE_NAME = "step"

logger = get_logger()


####################
# Helper functions #
####################


def get_translator(locale=None):
    from psynet.utils import get_translator

    return get_translator(
        locale, module=PACKAGE_NAME, locales_dir=os.path.join(here, "locales")
    )


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
DEFAULT_LOCALE = "en"


class StepCandidate:
    def __init__(self, text, previous_ratings=None, is_frozen=False):
        self.text = text
        self.hash = custom_hash(text)
        if previous_ratings is None:
            previous_ratings = []
        self.previous_ratings = previous_ratings
        self.is_frozen = is_frozen
        self.is_flagged = False


class StepStimulus:
    pass


class StepTagStimulus(StepStimulus):
    def __init__(self, url):
        self.url = url

    def preview(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def prompt(self, text, **kw):
        raise NotImplementedError("This method should be implemented in a subclass.")


class StepTagImage(StepTagStimulus):
    width = 350
    height = 350
    extensions = (".jpg", ".jpeg", ".png", ".gif")

    def preview(self):
        return f'<img src = "{self.url}" style="max-width:{self.width}px; width:100%; max-height="{self.height}">'

    def prompt(self, text, **kw):
        css = f"""
        <style>
        #prompt-image {{
            max-width: {self.width}px;
            max-height: {self.height}px;
        }}
        </style>
        """
        text = Markup(text + css)
        return ImagePrompt(self.url, text, width=None, height=None)


class StepTagAudioVisual(StepTagStimulus):
    max_height = 350

    def __init__(self, audio_url, img_url):
        self.audio_url = audio_url
        self.img_url = img_url
        super().__init__(audio_url)

    def preview(self):
        return f"""
        <img src = "{self.url}" style="width:100%; max-height="{self.max_height}">
        <audio controls>
            <source src="{self.audio_url}">
        </audio>
        """

    def prompt(self, text, **kw):
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
        prompt = f'<img src = "{self.img_url}" class="prompt_img"> <h3>{text}</h3>{css}'
        return AudioPrompt(
            audio=self.audio_url,
            text=Markup(prompt),
        )


class StepTagAudio(StepTagStimulus):
    controls = True
    extensions = (".mp3", ".wav")

    def preview(self):
        return f'<audio src = "{self.url}" controls>'

    def prompt(self, text, **kw):
        return AudioPrompt(self.url, text, controls=kw.get("controls", self.controls))


class StepTagVideo(StepTagStimulus):
    controls = False
    width = 300
    extensions = (".mp4", ".webm")

    def preview(self):
        return f'<video src = "{self.url}" style="max-width:{self.width}px; width:100%;" controls>'

    def prompt(self, text, **kw):
        return VideoPrompt(
            self.url,
            text,
            controls=kw.get("controls", self.controls),
            width=kw.get("width", self.width),
        )


def urls_to_start_nodes(urls):
    return [
        StepNode(
            definition=StepTagDefinition(
                stimulus=url_to_stimulus(url),
            )
        )
        for url in urls
    ]


def url_to_stimulus(url):
    if url.endswith(StepTagAudio.extensions):
        return StepTagAudio(url)
    elif url.endswith(StepTagVideo.extensions):
        return StepTagVideo(url)
    elif url.endswith(StepTagImage.extensions):
        return StepTagImage(url)
    else:
        raise ValueError(f"Unsupported file type: {url}")


def tags_to_candidates(tags: List[str]):
    assert sorted(tags) == sorted(set(tags)), "Duplicate tags found."
    return [StepCandidate(tag) for tag in tags]


class StepTagControl(Control):
    macro = "step_tag"
    external_template = "step.html"

    def __init__(self, frozen_candidates, unfrozen_candidates, n_stars, translations):
        super().__init__()
        self.frozen_candidates = frozen_candidates
        self.unfrozen_candidates = unfrozen_candidates
        self.n_stars = n_stars
        self.translations = translations

    @property
    def metadata(self):
        return self.__dict__


class StepDefinition:
    def __init__(self, stimulus: StepStimulus, candidates: List[StepCandidate] = None):
        if candidates is None:
            candidates = []
        self.stimulus = stimulus
        self.candidates = candidates


class StepTagDefinition(StepDefinition):
    def __init__(
        self,
        url: str = None,
        tags: List[str] = None,
        completed=False,
        stimulus=None,
        candidates=None,
    ):
        assert (
            sum([el is None for el in [url, stimulus]]) == 1
        ), "Specify either url or stimulus."
        assert (
            sum([el is not None for el in [tags, candidates]]) != 2
        ), "You cannot both specify tags and candidates."
        if stimulus is None:
            assert (
                url is not None
            ), "if there is no provided stimulus in StepTagDefinition, you must provide a non-empty url."
            stimulus = url_to_stimulus(url)
        if tags is None:
            tags = []
        if candidates is None:
            candidates = tags_to_candidates(tags)
        self.completed = completed
        super().__init__(stimulus, candidates)


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


class StepTagPage(StepPage):
    def __init__(
        self,
        stimulus: StepTagStimulus,
        frozen_candidates: List[StepCandidate],
        unfrozen_candidates: List[StepCandidate],
        available_tags: List[str],
        used_tags: List[str],
        n_stars: int,
        javascript_translations: dict,
        jinja_translations: dict,
        flagging_threshold: int = DEFAULT_FLAGGING_THRESHOLD,
        complete_on_n_frozen: int = 2,
        freeze_on_n_ratings: int = DEFAULT_FREEZE_ON_N_RATINGS,
        freeze_on_mean_rating: float = DEFAULT_FREEZE_ON_MEAN_RATING,
        **kwargs,
    ):
        self.stimulus = stimulus
        self.frozen_candidates = frozen_candidates
        self.unfrozen_candidates = unfrozen_candidates
        self.used_tags = used_tags

        super().__init__(
            flagging_threshold,
            complete_on_n_frozen,
            freeze_on_n_ratings,
            freeze_on_mean_rating,
            label="StepTagTrial",
            prompt=stimulus.prompt(text=""),
            control=StepTagControl(
                frozen_candidates=frozen_candidates,
                unfrozen_candidates=unfrozen_candidates,
                n_stars=n_stars,
                translations=jinja_translations,
            ),
            js_vars={
                **javascript_translations,
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
        gettext, pgettext = get_translator(self.var.get("locale"))

        return StepTagPage(
            stimulus=self.node.definition.stimulus,
            frozen_candidates=frozen_candidates,
            unfrozen_candidates=unfrozen_candidates,
            available_tags=available_tags,
            used_tags=used_tags,
            n_stars=self.trial_maker.n_stars,
            complete_on_n_frozen=self.trial_maker.complete_on_n_frozen,
            flagging_threshold=self.trial_maker.flagging_threshold,
            javascript_translations=self.trial_maker.get_javascript_translations(
                gettext, pgettext
            ),
            jinja_translations=self.trial_maker.get_jinja_translations(
                gettext, pgettext
            ),
        )


class StepNetwork(ImitationChainNetwork):
    pass


class StepNode(ImitationChainNode):
    time_estimate = Column(Integer)

    def create_initial_seed(self, experiment, participant):
        return {}

    def create_definition_from_seed(self, seed, experiment, participant):
        return seed

    def get_definitions(self):
        return [self.definition]

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

    locale: str
        The ISO-2 language code for the pipeline. The default is "en" (English).

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
        max_trials_per_participant: int = None,
        flagging_threshold: int = 2,
        n_stars: int = DEFAULT_N_STARS,
        locale: str = DEFAULT_LOCALE,
        freeze_on_n_ratings: int = 3,
        freeze_on_mean_rating: float = 3.0,
        complete_on_n_frozen: int = 1,
        rating_time_estimate: int = 1,
        creating_time_estimate: int = 4,
        view_time_estimate: int = 1,
        node_class=StepNode,
        debug=False,
        show_instructions=True,
        practice_stimuli: List[StepStimulus] = None,
        *args,
        **kwargs,
    ):
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
        assert (
            len(kwargs["start_nodes"]) >= expected_trials_per_participant
        ), "You must specify at least as many start nodes as trials per participant."
        kwargs["chains_per_experiment"] = len(kwargs["start_nodes"])

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
        time_estimates = [
            node.estimate_time(rating_time_estimate, creating_time_estimate, view_time_estimate)
            for node in kwargs["start_nodes"]
        ]

        self.mean_time_estimate = sum(time_estimates) / len(time_estimates)
        self.time_budget = (
            self.mean_time_estimate * self.expected_trials_per_participant
        )

        self.debug = debug
        self.show_instructions = show_instructions
        if practice_stimuli is None:
            practice_stimuli = []
        self.practice_stimuli = practice_stimuli
        self.n_practice_trials = len(practice_stimuli)
        self.locale = locale

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
            if candidate.head.estimate_time(
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
            rating = rating
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
            db.session.commit()

    def format_answer(self, trial, raw_answer):
        raise NotImplementedError("Must be implemented by subclass.")

    def finalize_trial(self, answer, trial, experiment, participant):
        answer = self.format_answer(trial, answer)
        trial.answer = answer
        super().finalize_trial(answer, trial, experiment, participant)
        self.decide_if_network_is_full(trial)

    @classmethod
    def get_instructions_before_practice(cls, locale, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")

    def get_practice(self):
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    def get_instructions_after_practice(cls, locale, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")

    @classmethod
    def get_instructions(cls, locale, debug=False, **kwargs):
        raise NotImplementedError("Must be implemented by subclass.")

    @property
    def introduction(self):
        pages = []

        if self.show_instructions:
            pages.append(self.get_instructions(self.locale))

        if self.n_practice_trials > 0:
            pages.append(self.get_instructions_before_practice(self.locale))
            pages.append(self.get_practice())
            pages.append(self.get_instructions_after_practice(self.locale))

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
        db.session.commit()


class StepTag(StepTrialMaker):
    response_timeout_sec = 5 * 60  # 5 minutes

    def __init__(
        self,
        vocabulary: List[str] = None,
        complete_on_n_frozen: int = 2,
        rating_time_estimate: int = 3,  # set based on pilot data
        creating_time_estimate: int = 6,  # set based on pilot data
        deposit_assets: bool = True,
        *args,
        **kwargs,
    ):
        if vocabulary is None:
            vocabulary = []
        self.vocabulary = vocabulary
        assets = []

        # Register assets
        if deposit_assets:
            for node in kwargs["start_nodes"]:
                url = node.definition.stimulus.url
                fname = basename(url)
                short_hash = hashlib.sha1(fname.encode()).hexdigest()
                # same filename is okay if they are from a different url
                local_key = f"{short_hash}_{fname}"
                asset = ExternalAsset(url=url, local_key=local_key)
                node.asset = asset
                assets.append(asset)

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

    def format_answer(self, trial, raw_answer):
        used_tags = trial.var.get("used_tags")
        frozen_candidates = trial.var.get("frozen_candidates")
        unfrozen_candidates = trial.var.get("unfrozen_candidates")
        hidden_candidates = trial.var.get("hidden_candidates")
        candidates = unfrozen_candidates + frozen_candidates + hidden_candidates
        used_tags = sorted(used_tags)
        shown_tags = [
            candidate.text
            for candidate in candidates
            if not (candidate.is_frozen or candidate.is_flagged)
        ]
        rated_tags = sorted(raw_answer["ratings"].keys())
        assert sorted(shown_tags) == rated_tags, "Tags must all be rated."

        new_tags = [html.unescape(tag) for tag in raw_answer["new_tags"] if tag not in used_tags]

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

        return StepTagDefinition(
            stimulus=trial.node.definition.stimulus,
            candidates=candidates,
            completed=n_frozen >= self.complete_on_n_frozen,
        )

    def prepare_trial(self, experiment, participant):
        trial, trial_status = super().prepare_trial(experiment, participant)
        if trial_status in ["wait", "exit"]:
            return trial, trial_status

        candidates = trial.definition.candidates

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
        trial.var.set("locale", self.locale)

        return trial, trial_status

    def create_networks_across(self, experiment):
        super().create_networks_across(experiment)
        initial_tags = [
            candidate.text
            for node in self.start_nodes
            for candidate in node.definition.candidates
        ]
        initial_vocabulary = list(set(initial_tags + self.vocabulary))
        Vocabulary.extend(initial_vocabulary)

    @staticmethod
    def get_instructions_without_tags(gettext, pgettext):
        _, _p = gettext, pgettext
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
    def get_instructions_with_tags(gettext, pgettext):
        _, _p = gettext, pgettext
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
    def get_jinja_translations(cls, gettext, pgettext):
        _, _p = gettext, pgettext
        return {
            "title_frozen": _p("STEP-Tag", "The following tags are already completed:"),
            "title_unfrozen": _p("STEP-Tag", "Mark the existing tags"),
            "type_more": _p("STEP-Tag", "Type more tags"),
            "next": _("Next"),
            "instructions_without_tags": cls.get_instructions_without_tags(
                gettext, pgettext
            ),
            "instructions_with_tags": cls.get_instructions_with_tags(gettext, pgettext),
        }

    @staticmethod
    def get_javascript_translations(gettext, pgettext):
        _, _p = gettext, pgettext
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
    def get_instructions_before_practice(cls, locale):
        _, _p = get_translator(locale)
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
                StepTagPractice(locale=self.locale, stimulus=stimulus, time_estimate=10)
            )
        return join(*practice_pages)

    @classmethod
    def get_instructions_after_practice(cls, locale, **kwargs):
        _, _p = get_translator(locale)
        return InfoPage(
            Markup(
                _p("STEP-Tag", "That was a practice round.")
                + " "
                + _p("STEP-Tag", "Now, let's start the main experiment!")
            ),
            time_estimate=3,
        )

    @classmethod
    def get_first_instructions(cls, locale):
        _, _p = get_translator(locale)
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
    def get_rating_instructions(cls, locale):
        _, _p = get_translator(locale)

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
    def get_creating_instructions(cls, locale):
        _, _p = get_translator(locale)

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
    def get_instructions(cls, locale, debug=False, **kwargs):
        _, _p = get_translator(locale)

        return join(
            cls.get_first_instructions(locale),
            cls.get_rating_instructions(locale),
            cls.get_creating_instructions(locale),
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
        locale: str = DEFAULT_LOCALE,
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
        gettext, pgettext = get_translator(locale)
        jinja_translations = step_class.get_jinja_translations(gettext, pgettext)
        javascript_translations = step_class.get_javascript_translations(
            gettext, pgettext
        )

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
