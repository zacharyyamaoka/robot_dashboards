"""Shared by the sim-to-real dashboards (sim2real_pendulum, sim2real_arm, sim2real_five_bar, sim2real_diff_drive).

Each one is the same question asked of a different robot: across many runs, which model predicts the real robot best,
and for how long? Zach's process (2026-11-25): calibrate so the virtual and real streams line up, run, compare every
sensor stream ("the more sensor streams the better"), then "a bunch of runs … analysis across those runs … drill down
and open each run". So every page is:

1. **Answer**: the models ranked against the reference (``real``) on one *ruler* stream (the sensor that reads the
   thing you care about most directly: a camera on the pendulum's link, the five-bar's tool, the base's pose): a
   score that favours short horizons, the error at each horizon, how long each model stays inside a tolerance that
   grows with time, and the headline run's ruler stream with its error against that envelope.
2. **Across runs**: robot_msg_viz's DatasetComparison summary (every stream, worst first, with the runs' tags).
3. **One tab per run**: its envelope; every stream compared (robot_msg_viz's RunComparison tabs); every stream
   recorded with all its sources overlaid (robot_msg_viz's Run view).

WHY robot_msg_viz's runs.py helpers (`log_run`, `run_layout`, `run_comparison_tabs`) and not its public `log` alone:
`viz.log(dataset_comparison)` draws the comparisons and the across-runs table, but a dashboard composes them with its
own answer and envelope into one page, and the helpers are the containers its own pages are built from. CI pins
robot_msg_viz to a commit (.github/workflows/site.yml), so a rename there fails this repo's tests before a user's.

The score and the envelope (on screen in every answer):

    tol(t) = floor + growth · t                                  the error a model may have t seconds into a run
    score  = Σ_h (1/h) · rms(0…h) / tol(h)  ÷  Σ_h (1/h)          h = 0.25, 0.5, 1, 2 s; below 1 = inside on average

WHY a tolerance that grows with time (Zach's note on sim-to-real): every model drifts from reality the longer it runs
(phase error, friction, chaos in more links), so a fixed bound fails every model at some horizon and says nothing about
the first second, which is where a model is used; and WHY 1/h weights: "evaluation should favour short horizons".
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import robot_msg_viz as viz
from robot_msg_viz import TIMELINE
from robot_msg_viz._engine import part
from robot_msg_viz.views.runs import headline, log_run, run_comparison_tabs, run_layout
from robot_msgs.compare import Comparison, Dataset, DatasetComparison, RunComparison, compare_dataset
from robot_msgs.compare.run_comparison import comparison_key, split_key
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory

from robot_dashboards.dashboards._motion_state_helpers import inner

DEG = 180.0 / math.pi
HORIZONS = (0.25, 0.5, 1.0, 2.0)
# per model, in order: the reference keeps robot_msg_viz's ink; the models a colour each, the tolerance a neutral grey
MODEL_COLOURS = ((95, 200, 120), (90, 150, 255), (255, 160, 60), (200, 120, 230))
REFERENCE_COLOUR = (230, 230, 235)
TOLERANCE_COLOUR = (150, 150, 155)
_COLOUR_NAMES = ("green", "blue", "orange", "purple")  # MODEL_COLOURS, in words for the view titles
RECORDED = "every stream recorded"  # a run's overlay tab; also its note's path (robot_msg_viz names both from it)


@dataclass(frozen=True)
class Envelope:
    """The error a model may have `t` seconds into a run: ``floor + growth · t`` (in `unit`)."""

    floor: float
    growth: float
    unit: str  # "°", "mm"

    def at(self, t: Any) -> np.ndarray:
        return self.floor + self.growth * np.asarray(t, dtype=np.float64)

    def amount(self, value: float) -> str:
        return f"{value:.3g}°" if self.unit == "°" else f"{value:.3g} {self.unit}"

    @property
    def formula(self) -> str:
        rate = f"{self.growth:g}°/s" if self.unit == "°" else f"{self.growth:g} {self.unit}/s"
        return f"tol(t) = {self.amount(self.floor)} + {rate} · t"


def joint_error_deg(comparison: Comparison) -> np.ndarray:
    """|error| of the worst joint at each sample, in degrees (a joint stream's positions)."""
    return np.max(np.abs(np.asarray(comparison.error.positions, dtype=np.float64)), axis=1) * DEG


def position_error_mm(comparison: Comparison) -> np.ndarray:
    """|error| of a pose stream's position at each sample, in mm."""
    return np.linalg.norm(np.asarray(comparison.error.positions, dtype=np.float64), axis=1) * 1000.0


@dataclass(frozen=True)
class Ruler:
    """The stream the models are ranked on, and how to read one comparison of it as |error| per sample."""

    stream: str
    label: str
    error: Callable[[Comparison], np.ndarray]
    envelope: Envelope
    reference: str = "real"
    horizons: tuple[float, ...] = HORIZONS


@dataclass(frozen=True)
class Score:
    """One model on one run, against the reference, on the ruler stream."""

    run: str
    model: str
    seconds: np.ndarray
    error: np.ndarray
    duration: float
    left_at: float | None  # the first t its error left the envelope; None: it never did
    rms: dict[float, float]
    inside: dict[float, bool]
    score: float

    @property
    def inside_for(self) -> float:
        return self.duration if self.left_at is None else self.left_at


def _comparison(result: RunComparison, a: str, b: str, stream: str) -> Comparison | None:
    for key in (comparison_key(a, b, stream), comparison_key(b, a, stream)):
        if key in result.comparisons:
            return result.comparisons[key]
    return None


def score_run(result: RunComparison, ruler: Ruler, models: tuple[str, ...]) -> list[Score]:
    out = []
    for model in models:
        comparison = _comparison(result, ruler.reference, model, ruler.stream)
        if comparison is None:
            continue
        seconds = np.asarray(comparison.error.t, dtype=np.float64)
        seconds = seconds - seconds[0]
        error = ruler.error(comparison)
        tol = ruler.envelope.at(seconds)
        outside = np.nonzero(error > tol)[0]
        rms, inside = {}, {}
        # WHY only the horizons the run lasts: a 2 s horizon on a 0.65 s run would be the 0.65 s number mislabelled
        reached = [h for h in ruler.horizons if h <= seconds[-1] + 1e-6] or [min(ruler.horizons)]
        for h in ruler.horizons:
            window = seconds <= h + 1e-9
            rms[h] = float(np.sqrt(np.mean(error[window] ** 2))) if h in reached else float("nan")
            inside[h] = bool(np.all(error[window] <= tol[window])) if h in reached else False
        weights = {h: 1.0 / h for h in reached}
        score = sum(weights[h] * rms[h] / float(ruler.envelope.at(h)) for h in reached) / sum(weights.values())
        out.append(Score(run=result.run, model=model, seconds=seconds, error=error, duration=float(seconds[-1]),
                         left_at=float(seconds[outside[0]]) if outside.size else None, rms=rms, inside=inside,
                         score=float(score)))  # fmt: skip
    return out


def _horizon(h: float) -> str:
    return f"{h:g} s"


def _for(score: Score, *, short: bool = False) -> str:
    if score.left_at is None:
        return f"≥ {score.duration:.2f} s" + ("" if short else " (never left)")
    return f"{score.left_at:.2f} s"


@dataclass(frozen=True)
class Sim2RealView:
    """One sim-to-real dashboard's Rerun renderer: ``log(rec, dataset=...)`` and its layout for a dataset."""

    key: str
    ruler: Ruler
    models: tuple[str, ...]
    headline: str  # the run the answer shows
    question: str
    sample: Callable[[], Dataset]
    overlay: tuple[str, ...] | None = None  # runs whose recorded streams are drawn (None: every run)
    paths: bool = True  # a pose ruler: draw each source's path for the headline run (from above)
    notes: str = ""
    extra_log: Callable[[Any, Dataset], None] | None = None  # dashboard-only views under the answer
    extra_views: Callable[[Dataset], list[Any]] = field(default=lambda dataset: [])

    # ── the numbers ────────────────────────────────────────────────────────────────────────────────────────────────

    def scores(self, result: DatasetComparison) -> dict[str, list[Score]]:
        return {name: score_run(run, self.ruler, self.models) for name, run in result.runs.items()}

    def _overlaid(self, dataset: Dataset) -> list[str]:
        names = [run.name for run in dataset.runs]
        return names if self.overlay is None else [name for name in names if name in self.overlay]

    def _headline(self, dataset: Dataset) -> str:
        names = [run.name for run in dataset.runs]
        return self.headline if self.headline in names else names[0]

    def answer(self, dataset: Dataset, result: DatasetComparison) -> str:
        scores = self.scores(result)
        ruler, envelope = self.ruler, self.ruler.envelope
        by_model: dict[str, list[Score]] = {}
        for found in scores.values():
            for score in found:
                by_model.setdefault(score.model, []).append(score)
        ranked = sorted(by_model.items(), key=lambda item: float(np.mean([s.score for s in item[1]])))
        if not ranked:
            return f"**{self.question}**\n\nNo run compares `{ruler.reference}` with a model on `{ruler.stream}`."
        runs = len(scores)

        def summary(found: list[Score]) -> tuple[float, str, int]:
            median = float(np.median([s.inside_for for s in found]))
            never = median >= min(s.duration for s in found) - 1e-9  # the median run never left the envelope
            return (float(np.mean([s.score for s in found])), ("≥ " if never else "") + f"{median:.2f} s",
                    sum(s.inside.get(1.0, False) for s in found))  # fmt: skip

        best, best_scores = ranked[0]
        score, inside_for, _ = summary(best_scores)
        others = ", ".join(f"{model} (score {summary(found)[0]:.2f}, inside {summary(found)[1]})"
                           for model, found in ranked[1:])  # fmt: skip
        lines = [
            f"### {self.question}",
            "",
            f"**{best} predicts `{ruler.reference}` best**: score **{score:.2f}**, inside the envelope for "
            f"**{inside_for}** (median of {runs} runs)" + (f". Then {others}." if others else "."),
            "",
            f"- **Ruler**: {ruler.label}, stream `{ruler.stream}`, each model against `{ruler.reference}`.",
            f"- **Envelope** (the error a model may have t seconds in): **{envelope.formula}**.",
            "- **Score** = Σ (1/h)·rms(0…h)/tol(h) ÷ Σ 1/h over h = "
            + ", ".join(_horizon(h) for h in ruler.horizons) + ": short horizons count most; below 1 = inside on "
            "average.",
            "",
            f"**Models, best first** (across {runs} runs)",
            "",
            "| # | model | score | inside for (median) | runs inside at 1 s |",
            "|---|---|---:|---:|---:|",
        ]  # fmt: skip
        for rank, (model, found) in enumerate(ranked, 1):
            s, t, at_one = summary(found)
            lines.append(f"| {rank} | **{model}** | {s:.2f} | {t} | {at_one} of {len(found)} |")
        lines += ["", (f"**Error at each horizon**: rms of |error| over 0…h ({envelope.unit}), mean of the runs that "
                       "last that long; ✓ every one stayed inside the envelope up to h"), "",
                  "| model | " + " | ".join(f"{_horizon(h)} (tol {envelope.amount(float(envelope.at(h)))})"
                                            for h in ruler.horizons) + " |",
                  "|---|" + "---:|" * len(ruler.horizons)]  # fmt: skip
        for model, found in ranked:
            cells = []
            for h in ruler.horizons:
                lasted = [s for s in found if np.isfinite(s.rms[h])]
                if not lasted:
                    cells.append("—")
                    continue
                mean = float(np.mean([s.rms[h] for s in lasted]))
                mark = " ✓" if all(s.inside[h] for s in lasted) else ""
                cells.append(f"{envelope.amount(mean)}{mark}" + (f" ({len(lasted)} runs)" if len(lasted) < len(found)
                                                                  else ""))  # fmt: skip
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        models = [model for model, _ in ranked]
        lines += self._streams_table(result, models)
        tag_keys = list(dict.fromkeys(key for run in dataset.runs for key in run.tags))
        lines += ["", "**Each run**: score · inside for (open the run's tab to drill in)", "",
                  "| run | " + " | ".join(tag_keys) + " | " + " | ".join(models) + " |",
                  "|---|" + "---|" * len(tag_keys) + "---:|" * len(models)]  # fmt: skip
        for run in dataset.runs:
            found = {s.model: s for s in scores.get(run.name, [])}
            cells = [f"{found[m].score:.2f} · {_for(found[m], short=True)}" if m in found else "—" for m in models]
            lines.append(f"| {run.name} | " + " | ".join(str(run.tags.get(k, "—")) for k in tag_keys) + " | "
                         + " | ".join(cells) + " |")  # fmt: skip
        if self.notes:
            lines += ["", f"*{self.notes}*"]
        return "\n".join(lines)

    def _streams_table(self, result: DatasetComparison, models: list[str]) -> list[str]:
        """Every stream: each model's worst run against the reference, by robot_msg_viz's headline number."""
        worst: dict[str, dict[str, tuple[float, str, str, str]]] = {}
        for name, run in result.runs.items():
            for model in models:
                for key in run.comparisons:
                    a, b, stream = split_key(key)
                    if {a, b} != {self.ruler.reference, model}:
                        continue
                    what, value, unit, _ = headline(run.comparisons[key])
                    best = worst.setdefault(stream, {}).get(model)
                    if np.isfinite(value) and (best is None or value > best[0]):
                        worst[stream][model] = (value, unit, name, what.split(",")[0])
        if not worst:
            return []
        lines = ["", (f"**Every stream**: each model's worst run against `{self.ruler.reference}` (drill in: Across "
                      "runs, then the run's tab)"), "", "| stream | measure | " + " | ".join(models) + " |",
                 "|---|---|" + "---:|" * len(models)]  # fmt: skip
        for stream, by_model in worst.items():
            what = next(iter(by_model.values()))[3]
            cells = []
            for model in models:
                if model in by_model:
                    value, unit, run_name, _ = by_model[model]
                    cells.append(f"{value:.3g} {unit} ({run_name})")
                else:
                    cells.append("—")
            lines.append(f"| {stream} | {what} | " + " | ".join(cells) + " |")
        return lines

    def run_note(self, run: Any, found: list[Score]) -> str:
        ruler, envelope = self.ruler, self.ruler.envelope
        lines = [f"**{run.title or run.name}** · {ruler.label} against `{ruler.reference}` · {envelope.formula}", ""]
        if run.tags:
            lines += [" · ".join(f"{key} = {value}" for key, value in run.tags.items()), ""]
        offsets = [f"{source} {value * 1000:.1f} ms" for source, value in run.time_offset_s.items() if value]
        if offsets:
            lines += ["Calibrated clock offset: " + ", ".join(offsets), ""]
        lines += ["| model | score | inside for | " + " | ".join(f"rms 0…{_horizon(h)}" for h in ruler.horizons)
                  + " |", "|---|---:|---:|" + "---:|" * len(ruler.horizons)]  # fmt: skip
        for score in sorted(found, key=lambda s: s.score):
            cells = [(envelope.amount(score.rms[h]) + (" ✓" if score.inside[h] else " ✗")) if np.isfinite(score.rms[h])
                     else "— (run ends)" for h in ruler.horizons]  # fmt: skip
            lines.append(f"| {score.model} | {score.score:.2f} | {_for(score)} | " + " | ".join(cells) + " |")
        lines += ["", ("*✓ inside the envelope all the way to h, ✗ left it before h. The other tabs: every stream "
                       "compared, every stream recorded with all sources overlaid.*")]  # fmt: skip
        return "\n".join(lines)

    # ── logging ────────────────────────────────────────────────────────────────────────────────────────────────────

    def _log_envelope(self, stream: Any, run: str, found: list[Score]) -> None:
        envelope = self.ruler.envelope
        if not found:
            return
        longest = max(found, key=lambda s: s.duration)
        root = f"{self.key}/envelope/{part(run)}"
        unit = envelope.unit
        series = [("tolerance", f"tol(t) ({unit})", TOLERANCE_COLOUR, 3.0, longest.seconds,
                   envelope.at(longest.seconds))]  # fmt: skip
        for score in found:
            colour = MODEL_COLOURS[self.models.index(score.model) % len(MODEL_COLOURS)]
            series.append((score.model, f"|error| {score.model} ({unit})", colour, 2.0, score.seconds, score.error))
        for name, label, colour, width, seconds, values in series:
            path = f"{root}/{part(name)}"
            stream.log(path, rr.SeriesLines(names=[label], colors=[colour], widths=[width]), static=True)
            stream.send_columns(path, indexes=[rr.TimeColumn(TIMELINE, duration=seconds)],
                                columns=rr.Scalars.columns(scalars=np.asarray(values, dtype=np.float64)))  # fmt: skip

    def _log_paths(self, stream: Any, dataset: Dataset) -> None:
        """The headline run's ruler path per source, when the ruler is a pose (top view of where each went)."""
        run = next(r for r in dataset.runs if r.name == self._headline(dataset))
        for index, (source, recording) in enumerate(run.recordings.items()):
            msg = recording.streams.get(self.ruler.stream)
            if not isinstance(msg, PoseTrajectory):
                continue
            colour = REFERENCE_COLOUR if source == self.ruler.reference else (
                MODEL_COLOURS[self.models.index(source) % len(MODEL_COLOURS)] if source in self.models else (160, 160, 160))
            stream.log(f"{self.key}/paths/{part(source)}",
                       rr.LineStrips3D([np.asarray(msg.positions, dtype=np.float64)], colors=[colour],
                                       radii=rr.Radius.ui_points(2.5 if index == 0 else 1.8)),
                       static=True)  # fmt: skip

    def log(self, recording: Any, *, dataset: Dataset | None = None) -> None:
        if dataset is None:
            return
        stream = inner(recording)
        result = _compared(dataset)
        if dataset is not self.sample():
            # WHY only off the sample: the layout has a tab per run, and the fixed blueprint is the sample's
            stream.send_blueprint(self.blueprint(dataset), make_active=True, make_default=True)
        viz.log(result, self.key, recording=stream)  # every comparison + the across-runs table (robot_msg_viz)
        for run in dataset.runs:
            if run.name in self._overlaid(dataset):
                log_run(run, stream, base=f"runs/{part(run.name)}/", name=RECORDED)
        scores = self.scores(result)
        for run in dataset.runs:
            found = scores.get(run.name, [])
            self._log_envelope(stream, run.name, found)
            stream.log(f"{self.key}/runs/{part(run.name)}", rr.TextDocument(self.run_note(run, found),
                                                                            media_type="text/markdown"), static=True)
        if self.paths:
            self._log_paths(stream, dataset)
        stream.log(f"{self.key}/answer", rr.TextDocument(self.answer(dataset, result), media_type="text/markdown"),
                   static=True)  # fmt: skip
        if self.extra_log is not None:
            self.extra_log(stream, dataset)

    # ── the layout ─────────────────────────────────────────────────────────────────────────────────────────────────

    def _envelope_view(self, run: str, title: str, duration: float) -> rrb.View:
        # WHY a fixed y range (1.5 × the envelope at the end): a model far outside (the analytical pendulum's 40°)
        # flattened the models that matter onto the axis; leaving the top of the plot IS the answer for that one
        top = 1.5 * float(self.ruler.envelope.at(duration))
        return rrb.TimeSeriesView(origin=f"/{self.key}/envelope/{part(run)}", name=title,
                                  plot_legend=rrb.PlotLegend(corner="LeftTop"),
                                  axis_y=rrb.ScalarAxis(range=(0.0, top)))  # fmt: skip

    def blueprint(self, dataset: Dataset) -> rrb.Blueprint:
        result = _compared(dataset)
        ruler, envelope = self.ruler, self.ruler.envelope
        headline = self._headline(dataset)
        first = next(r for r in dataset.runs if r.name == headline)
        recorded = first.recordings.get(ruler.reference, next(iter(first.recordings.values()))).streams.get(ruler.stream)
        right: list[Any] = [self._envelope_view(
            headline, f"{headline} · |{ruler.stream} error| ({envelope.unit}) per model vs {envelope.formula}",
            _duration(first))]
        if isinstance(recorded, PoseTrajectory) and self.paths:
            colours = ", ".join(f"{model} {_COLOUR_NAMES[self.models.index(model) % len(_COLOUR_NAMES)]}"
                                for model in self.models)  # fmt: skip
            right.append(rrb.Spatial3DView(origin=f"/{self.key}/paths", eye_controls=_from_above(first, ruler.stream),
                                           name=f"{headline} · {ruler.stream} from above (m): {ruler.reference} white, "
                                                f"{colours}"))  # fmt: skip
        elif isinstance(recorded, JointTrajectory) and headline in self._overlaid(dataset):
            right.append(rrb.TimeSeriesView(
                origin=f"/runs/{part(headline)}/streams/{part(ruler.stream)}/positions",
                name=f"{headline} · {ruler.stream} (°), every source overlaid: {ruler.reference} first, the models "
                     "lighter / darker shades", plot_legend=rrb.PlotLegend(corner="LeftTop")))  # fmt: skip
        right += self.extra_views(dataset)
        answer = rrb.Horizontal(
            rrb.TextDocumentView(origin=f"/{self.key}/answer", name="Answer"),
            rrb.Vertical(*right), column_shares=[1.15, 1], name="Answer",
        )  # fmt: skip
        across = rrb.TextDocumentView(origin=f"/notes/{part(self.key)}", name="Across runs · every stream")
        tabs = []
        for run in dataset.runs:
            base = f"runs/{part(run.name)}/"
            inner_tabs = [
                rrb.Horizontal(self._envelope_view(run.name, f"|{ruler.stream} error| ({envelope.unit}) vs "
                                                   f"{envelope.formula}", _duration(run)),
                               rrb.TextDocumentView(origin=f"/{self.key}/runs/{part(run.name)}", name=run.name),
                               column_shares=[3, 2], name="envelope"),
                rrb.Horizontal(rrb.Tabs(*run_comparison_tabs(result.runs[run.name])),
                               rrb.TextDocumentView(origin=f"/{base}notes/{part(run.name)}", name=f"run {run.name}"),
                               column_shares=[3, 2], name="every stream compared"),
            ]  # fmt: skip
            if run.name in self._overlaid(dataset):
                inner_tabs.append(run_layout(run, base, RECORDED))
            tabs.append(rrb.Tabs(*inner_tabs, name=run.name))
        return rrb.Blueprint(
            rrb.Tabs(answer, across, *tabs, active_tab=0),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed", timeline=TIMELINE),
            collapse_panels=False,
        )

    def sample_blueprint(self) -> rrb.Blueprint:
        return _blueprint_of(self, self.sample())


def _from_above(run: Any, stream: str) -> rrb.EyeControls3D:
    """A camera above the middle of every source's path, tilted a little so the ground plane still reads as one."""
    points = np.vstack([np.asarray(rec.streams[stream].positions, dtype=np.float64) for rec in run.recordings.values()
                        if isinstance(rec.streams.get(stream), PoseTrajectory)])  # fmt: skip
    low, high = points.min(axis=0), points.max(axis=0)
    middle, span = (low + high) / 2.0, max(float(np.max(high - low)), 0.2)
    return rrb.EyeControls3D(kind="Orbital", look_target=middle.tolist(), eye_up=[0.0, 0.0, 1.0],
                             position=(middle + [0.0, -0.25 * span, 0.95 * span]).tolist())  # fmt: skip


def _duration(run: Any) -> float:
    spans = [float(msg.duration) for rec in run.recordings.values() for msg in rec.streams.values()
             if hasattr(msg, "duration")]  # fmt: skip
    return max(spans, default=1.0)


@functools.cache
def _blueprint_of(view: Sim2RealView, dataset: Dataset) -> rrb.Blueprint:
    return view.blueprint(dataset)


# WHY cached (a Dataset hashes by identity): log and blueprint both need the comparison, and the sample is one object
@functools.lru_cache(maxsize=8)
def _compared(dataset: Dataset) -> DatasetComparison:
    return compare_dataset(dataset)
