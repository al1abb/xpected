"""Ties elo.py + dixon_coles.py + league_strength.py into one prediction per
match: a (lambda_home, lambda_away, rho) triple turned into a full scoreline
matrix, then reduced to 1X2 / over-under / BTTS / top scorelines.

Per-match lambda source:
- Both teams have enough history in a shared domestic-league Dixon-Coles fit:
  use it directly (richest signal — real shots/goals in that specific league
  environment).
- Otherwise (cross-league UEFA fixture, Azerbaijan Premyer Liqa, or a
  thin-history team): fall back to the Elo-bridge, blended in proportion to
  how much domestic history exists — a team with only 3 matches in its
  league's fit gets pulled mostly toward what Elo alone implies, per the
  plan's shrinkage design.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sqlalchemy.orm import Session

from app.config import COMPETITIONS
from app.models import Competition, Match, ModelRun, Prediction
from model import dixon_coles, elo, league_strength

MAX_GOALS = 10
SHRINKAGE_MATCH_THRESHOLD = 10
LOW_CONFIDENCE_WEIGHT = 0.5
# Shots-on-target are a lower-variance proxy for chance quality than goals
# (a rough stand-in for xG, which we don't have) — blending a shots-based fit
# into the goals-based one gives a steadier read on team strength, especially
# early in a season when goal counts alone are still noisy. Kept deliberately
# modest: shots predict *shots*, not goals directly, so this is a refinement
# on top of the goals fit, never a replacement for it.
SHOTS_BLEND_WEIGHT = 0.25
# For cross-league matches (no shared Dixon-Coles pool — every UEFA fixture,
# by construction), confidence is judged on *overall* match history across
# every competition, not on whether a domestic-league fit exists at all —
# otherwise every single UEFA match gets flagged low regardless of how
# established either club is, which is true by construction and therefore
# useless as a signal. Real Madrid vs Inter should not read the same as two
# sides making their European debut.
CROSS_LEAGUE_CONFIDENCE_THRESHOLD = 15

MAIN_LEAGUE_COMPETITION_SLUGS = [c["slug"] for c in COMPETITIONS if c["fd_code"]]


def score_matrix(lambda_home: float, lambda_away: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    from scipy.stats import poisson

    goals = np.arange(max_goals + 1)
    home_probs = poisson.pmf(goals, lambda_home)
    away_probs = poisson.pmf(goals, lambda_away)
    matrix = np.outer(home_probs, away_probs)

    hg, ag = np.meshgrid(goals, goals, indexing="ij")
    lam_grid = np.full_like(matrix, lambda_home)
    mu_grid = np.full_like(matrix, lambda_away)
    matrix = matrix * dixon_coles.tau(hg.astype(float), ag.astype(float), lam_grid, mu_grid, rho)

    total = matrix.sum()
    return matrix / total if total > 0 else matrix


def summarize_matrix(matrix: np.ndarray) -> dict:
    n = matrix.shape[0]
    idx = np.arange(n)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")

    home_win = float(matrix[ii > jj].sum())
    draw = float(matrix[ii == jj].sum())
    away_win = float(matrix[ii < jj].sum())
    over_2_5 = float(matrix[(ii + jj) >= 3].sum())
    btts = float(matrix[(ii >= 1) & (jj >= 1)].sum())

    flat_order = np.argsort(matrix, axis=None)[::-1][:5]
    top_scorelines = [{"score": f"{i}-{j}", "prob": float(matrix[i, j])} for i, j in (divmod(int(f), n) for f in flat_order)]

    return {
        "home_win_prob": home_win,
        "draw_prob": draw,
        "away_win_prob": away_win,
        "over_2_5_prob": over_2_5,
        "btts_prob": btts,
        "top_scorelines": top_scorelines,
    }


class Predictor:
    def __init__(self, session: Session, *, as_of: dt.datetime | None = None, exclude_match_id: int | None = None):
        self.session = session
        self.as_of = as_of or dt.datetime.utcnow()
        self.elo_ratings = elo.compute_ratings(session, as_of=as_of, exclude_match_id=exclude_match_id)
        self.overall_match_counts = elo.match_count_by_team(session, as_of=as_of)
        self.dc_fits = self._fit_all_leagues(exclude_match_id)
        self.shots_fits, self.conversion_rates = self._fit_shots_pools(exclude_match_id)
        self.elo_calib = league_strength.fit(session, self.elo_ratings)

    def _fit_all_leagues(self, exclude_match_id: int | None) -> dict[int, dixon_coles.LeagueFit]:
        fits = {}
        for slug in MAIN_LEAGUE_COMPETITION_SLUGS:
            competition = self.session.query(Competition).filter_by(slug=slug).one_or_none()
            if competition is None:
                continue
            query = self.session.query(Match).filter_by(competition_id=competition.id, status="finished")
            if self.as_of is not None:
                query = query.filter(Match.utc_kickoff <= self.as_of)
            rows = [
                {
                    "home_team_id": m.home_team_id,
                    "away_team_id": m.away_team_id,
                    "home_goals": m.home_goals,
                    "away_goals": m.away_goals,
                    "utc_kickoff": m.utc_kickoff,
                }
                for m in query.all()
                if m.id != exclude_match_id
            ]
            try:
                fits[competition.id] = dixon_coles.fit_league(rows, as_of=self.as_of)
            except ValueError:
                continue
        return fits

    def _fit_shots_pools(
        self, exclude_match_id: int | None
    ) -> tuple[dict[int, dixon_coles.LeagueFit], dict[int, float]]:
        """Same per-league fit machinery as _fit_all_leagues, but on shots-on-
        target instead of goals — only for matches where that data exists
        (football-data.co.uk sourced). Also returns each league's
        goals-per-shot-on-target conversion rate, needed to rescale the
        shots-fit's implied rate back onto a goals-like scale for blending."""
        fits: dict[int, dixon_coles.LeagueFit] = {}
        rates: dict[int, float] = {}
        for slug in MAIN_LEAGUE_COMPETITION_SLUGS:
            competition = self.session.query(Competition).filter_by(slug=slug).one_or_none()
            if competition is None:
                continue
            query = self.session.query(Match).filter_by(competition_id=competition.id, status="finished")
            if self.as_of is not None:
                query = query.filter(Match.utc_kickoff <= self.as_of)
            matches = [
                m
                for m in query.all()
                if m.id != exclude_match_id and m.home_shots_on_target is not None and m.away_shots_on_target is not None
            ]
            if not matches:
                continue

            total_goals = sum(m.home_goals + m.away_goals for m in matches)
            total_shots_on_target = sum(m.home_shots_on_target + m.away_shots_on_target for m in matches)
            if total_shots_on_target == 0:
                continue
            rates[competition.id] = total_goals / total_shots_on_target

            rows = [
                {
                    "home_team_id": m.home_team_id,
                    "away_team_id": m.away_team_id,
                    "home_goals": m.home_shots_on_target,
                    "away_goals": m.away_shots_on_target,
                    "utc_kickoff": m.utc_kickoff,
                }
                for m in matches
            ]
            try:
                fits[competition.id] = dixon_coles.fit_league(rows, as_of=self.as_of)
            except ValueError:
                continue
        return fits, rates

    def _avg_rho(self) -> float:
        rhos = [f.rho for f in self.dc_fits.values()]
        return float(np.mean(rhos)) if rhos else 0.0

    def lambdas_for(self, match: Match) -> tuple[float, float, float, float]:
        """Returns (lambda_home, lambda_away, rho, confidence_weight).

        For a same-league match: confidence_weight is also the Dixon-Coles/
        Elo blend ratio (1.0 = fully-trusted domestic fit, shrinking toward
        Elo for thin-history teams).

        For a cross-league match (no shared Dixon-Coles pool — every UEFA
        fixture): lambdas are always pure Elo-bridge, and confidence_weight
        instead reflects how much *overall* history each team has across any
        competition, so an established club's European fixture doesn't get
        flagged low just because it's cross-league by construction.
        """
        elo_home = self.elo_ratings.get(match.home_team_id, elo.BASE_RATING)
        elo_away = self.elo_ratings.get(match.away_team_id, elo.BASE_RATING)
        elo_lh, elo_la = self.elo_calib.lambdas(elo_home, elo_away)

        fit = self.dc_fits.get(match.competition_id)
        dc_result = fit.lambdas(match.home_team_id, match.away_team_id) if fit else None
        if dc_result is None:
            n_home = self.overall_match_counts.get(match.home_team_id, 0)
            n_away = self.overall_match_counts.get(match.away_team_id, 0)
            elo_confidence = min(1.0, min(n_home, n_away) / CROSS_LEAGUE_CONFIDENCE_THRESHOLD)
            return elo_lh, elo_la, self._avg_rho(), elo_confidence

        dc_lh, dc_la = dc_result
        dc_lh, dc_la = self._blend_in_shots(match, dc_lh, dc_la)

        n_home = fit.match_counts.get(match.home_team_id, 0)
        n_away = fit.match_counts.get(match.away_team_id, 0)
        weight = min(1.0, min(n_home, n_away) / SHRINKAGE_MATCH_THRESHOLD)

        lambda_home = weight * dc_lh + (1 - weight) * elo_lh
        lambda_away = weight * dc_la + (1 - weight) * elo_la
        return lambda_home, lambda_away, fit.rho, weight

    def _blend_in_shots(self, match: Match, dc_lh: float, dc_la: float) -> tuple[float, float]:
        shots_fit = self.shots_fits.get(match.competition_id)
        rate = self.conversion_rates.get(match.competition_id)
        if shots_fit is None or rate is None:
            return dc_lh, dc_la

        shots_result = shots_fit.lambdas(match.home_team_id, match.away_team_id)
        if shots_result is None:
            return dc_lh, dc_la

        shots_lh, shots_la = shots_result
        blended_home = (1 - SHOTS_BLEND_WEIGHT) * dc_lh + SHOTS_BLEND_WEIGHT * (shots_lh * rate)
        blended_away = (1 - SHOTS_BLEND_WEIGHT) * dc_la + SHOTS_BLEND_WEIGHT * (shots_la * rate)
        return blended_home, blended_away

    def predict_match(self, match: Match) -> dict:
        lambda_home, lambda_away, rho, weight = self.lambdas_for(match)
        matrix = score_matrix(lambda_home, lambda_away, rho)
        summary = summarize_matrix(matrix)
        summary["home_xg_pred"] = lambda_home
        summary["away_xg_pred"] = lambda_away
        summary["confidence"] = "low" if weight < LOW_CONFIDENCE_WEIGHT else "normal"
        return summary


def generate_predictions(session: Session, *, notes: str | None = None) -> int:
    """Fits the model against everything finished so far and predicts every
    currently-scheduled match. Each call creates a new ModelRun, so past
    predictions stay attributable to the model state that produced them —
    required for /accuracy to score predictions without hindsight bias."""
    predictor = Predictor(session)

    model_run = ModelRun(
        params={
            "xi_per_day": dixon_coles.XI_PER_DAY,
            "shrinkage_threshold": SHRINKAGE_MATCH_THRESHOLD,
            "elo_base": elo.BASE_RATING,
            "elo_k": elo.K_BASE,
            "elo_home_advantage": elo.HOME_ADVANTAGE,
            "elo_calibration_slope": predictor.elo_calib.slope,
            "elo_calibration_intercept": predictor.elo_calib.intercept,
            "shots_blend_weight": SHOTS_BLEND_WEIGHT,
        },
        notes=notes,
    )
    session.add(model_run)
    session.flush()

    scheduled = session.query(Match).filter_by(status="scheduled").all()
    count = 0
    for match in scheduled:
        summary = predictor.predict_match(match)
        session.add(
            Prediction(
                match_id=match.id,
                model_run_id=model_run.id,
                home_win_prob=summary["home_win_prob"],
                draw_prob=summary["draw_prob"],
                away_win_prob=summary["away_win_prob"],
                home_xg_pred=summary["home_xg_pred"],
                away_xg_pred=summary["away_xg_pred"],
                over_2_5_prob=summary["over_2_5_prob"],
                btts_prob=summary["btts_prob"],
                top_scorelines=summary["top_scorelines"],
                confidence=summary["confidence"],
            )
        )
        count += 1

    session.commit()
    return count
