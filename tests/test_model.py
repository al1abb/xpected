import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Competition, Match, Team
from model import dixon_coles, elo, league_strength, predict


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


# ---------- elo.py ----------


def test_elo_is_zero_sum_across_all_teams(session):
    teams = [Team(canonical_name=f"Team {i}") for i in range(4)]
    session.add_all(teams)
    session.flush()

    base = dt.datetime(2026, 1, 1)
    session.add_all(
        [
            Match(competition_id=1, utc_kickoff=base, status="finished", home_team_id=teams[0].id, away_team_id=teams[1].id, home_goals=3, away_goals=0, source="test"),
            Match(competition_id=1, utc_kickoff=base + dt.timedelta(days=1), status="finished", home_team_id=teams[2].id, away_team_id=teams[3].id, home_goals=1, away_goals=1, source="test"),
            Match(competition_id=1, utc_kickoff=base + dt.timedelta(days=2), status="finished", home_team_id=teams[1].id, away_team_id=teams[2].id, home_goals=0, away_goals=2, source="test"),
        ]
    )
    session.commit()

    ratings = elo.compute_ratings(session)
    assert len(ratings) == 4
    total = sum(ratings.values()) + (4 - len(ratings)) * elo.BASE_RATING
    assert total == pytest.approx(4 * elo.BASE_RATING, abs=1e-6)


def test_elo_winner_gains_loser_loses(session):
    teams = [Team(canonical_name="Strong"), Team(canonical_name="Weak")]
    session.add_all(teams)
    session.flush()
    session.add(
        Match(competition_id=1, utc_kickoff=dt.datetime(2026, 1, 1), status="finished", home_team_id=teams[0].id, away_team_id=teams[1].id, home_goals=2, away_goals=0, source="test")
    )
    session.commit()

    ratings = elo.compute_ratings(session)
    assert ratings[teams[0].id] > elo.BASE_RATING
    assert ratings[teams[1].id] < elo.BASE_RATING


def test_bigger_margin_moves_rating_more():
    def rating_after(home_goals, away_goals):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        s = sessionmaker(bind=engine)()
        t1, t2 = Team(canonical_name="A"), Team(canonical_name="B")
        s.add_all([t1, t2])
        s.flush()
        t1_id = t1.id
        s.add(Match(competition_id=1, utc_kickoff=dt.datetime(2026, 1, 1), status="finished", home_team_id=t1.id, away_team_id=t2.id, home_goals=home_goals, away_goals=away_goals, source="test"))
        s.commit()
        r = elo.compute_ratings(s)
        s.close()
        return r[t1_id]

    small_win = rating_after(1, 0)
    big_win = rating_after(4, 0)
    assert big_win > small_win


# ---------- dixon_coles.py ----------


def _synthetic_league_matches(n_teams=6, matches_per_pair=3, strong_idx=0, weak_idx=1, as_of=None):
    as_of = as_of or dt.datetime(2026, 1, 1)
    rows = []
    rng = np.random.default_rng(42)
    team_ids = list(range(n_teams))
    for i in team_ids:
        for j in team_ids:
            if i == j:
                continue
            for k in range(matches_per_pair):
                home_lambda = 2.0 if i == strong_idx else (0.5 if i == weak_idx else 1.3)
                away_lambda = 0.4 if j == strong_idx else (1.0 if j == weak_idx else 0.9)
                rows.append(
                    {
                        "home_team_id": i,
                        "away_team_id": j,
                        "home_goals": int(rng.poisson(home_lambda)),
                        "away_goals": int(rng.poisson(away_lambda)),
                        "utc_kickoff": as_of - dt.timedelta(days=int(rng.integers(1, 300))),
                    }
                )
    return rows


def test_dixon_coles_fits_strong_team_higher_attack():
    rows = _synthetic_league_matches()
    fit = dixon_coles.fit_league(rows, as_of=dt.datetime(2026, 1, 1))
    assert fit.attack[0] > fit.attack[1]  # strong_idx=0 should out-attack weak_idx=1


def test_dixon_coles_attack_sums_to_zero():
    rows = _synthetic_league_matches()
    fit = dixon_coles.fit_league(rows, as_of=dt.datetime(2026, 1, 1))
    assert sum(fit.attack.values()) == pytest.approx(0.0, abs=1e-6)


def test_dixon_coles_lambdas_none_for_unknown_team():
    rows = _synthetic_league_matches()
    fit = dixon_coles.fit_league(rows, as_of=dt.datetime(2026, 1, 1))
    assert fit.lambdas(0, 9999) is None


def test_dixon_coles_raises_on_too_little_data():
    with pytest.raises(ValueError):
        dixon_coles.fit_league([{"home_team_id": 1, "away_team_id": 2, "home_goals": 1, "away_goals": 0, "utc_kickoff": dt.datetime(2026, 1, 1)}], as_of=dt.datetime(2026, 1, 1))


# ---------- league_strength.py ----------


def test_elo_calibration_positive_slope(session):
    comp = Competition(slug="champions-league", name="UCL", country="Europe", type="uefa_cup")
    session.add(comp)
    session.flush()

    teams = [Team(canonical_name=f"T{i}") for i in range(6)]
    session.add_all(teams)
    session.flush()

    elo_ratings = {teams[i].id: 1500 + (i - 2.5) * 150 for i in range(6)}

    rng = np.random.default_rng(1)
    matches = []
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            diff = elo_ratings[teams[i].id] - elo_ratings[teams[j].id]
            expected_gd = diff / 200
            hg = max(0, int(round(rng.poisson(max(0.3, 1.3 + expected_gd / 2)))))
            ag = max(0, int(round(rng.poisson(max(0.3, 1.3 - expected_gd / 2)))))
            matches.append(Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 1, 1), status="finished", home_team_id=teams[i].id, away_team_id=teams[j].id, home_goals=hg, away_goals=ag, source="test"))
    session.add_all(matches)
    session.commit()

    calib = league_strength.fit(session, elo_ratings)
    assert calib.slope > 0


def test_elo_calibration_falls_back_with_too_few_matches(session):
    calib = league_strength.fit(session, {})
    assert calib.slope == pytest.approx(1.0 / 100)
    assert calib.avg_total_goals == pytest.approx(2.6)


# ---------- predict.py ----------


def test_score_matrix_sums_to_one():
    matrix = predict.score_matrix(1.4, 1.1, -0.05)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-9)


def test_score_matrix_rho_changes_low_scores():
    neutral = predict.score_matrix(1.2, 1.2, 0.0)
    corrected = predict.score_matrix(1.2, 1.2, -0.1)
    assert neutral[0, 0] != pytest.approx(corrected[0, 0])


def test_summarize_matrix_probabilities_sum_to_one():
    matrix = predict.score_matrix(1.5, 1.0, -0.05)
    summary = predict.summarize_matrix(matrix)
    total = summary["home_win_prob"] + summary["draw_prob"] + summary["away_win_prob"]
    assert total == pytest.approx(1.0, abs=1e-6)
    assert 0 <= summary["over_2_5_prob"] <= 1
    assert 0 <= summary["btts_prob"] <= 1
    assert len(summary["top_scorelines"]) == 5


def test_stronger_team_favoured_in_summary():
    matrix = predict.score_matrix(2.2, 0.7, -0.05)
    summary = predict.summarize_matrix(matrix)
    assert summary["home_win_prob"] > summary["away_win_prob"]


def test_predictor_uses_dixon_coles_for_established_domestic_team(session):
    comp = Competition(slug="premier-league", name="EPL", country="England", type="league", fd_code="E0")
    session.add(comp)
    session.flush()

    rows = _synthetic_league_matches(as_of=dt.datetime(2026, 6, 1))
    teams = {i: Team(canonical_name=f"Team {i}") for i in range(6)}
    session.add_all(teams.values())
    session.flush()

    matches = [
        Match(competition_id=comp.id, utc_kickoff=r["utc_kickoff"], status="finished", home_team_id=teams[r["home_team_id"]].id, away_team_id=teams[r["away_team_id"]].id, home_goals=r["home_goals"], away_goals=r["away_goals"], source="test")
        for r in rows
    ]
    session.add_all(matches)
    session.commit()

    fixture = Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 6, 2), status="scheduled", home_team_id=teams[0].id, away_team_id=teams[1].id, source="test")
    session.add(fixture)
    session.commit()

    predictor = predict.Predictor(session, as_of=dt.datetime(2026, 6, 2))
    _, _, _, weight = predictor.lambdas_for(fixture)
    assert weight == 1.0  # plenty of matches for both teams -> full trust in Dixon-Coles


def test_established_teams_get_normal_confidence_in_cross_league_match(session):
    """Regression: every UEFA fixture used to get flagged 'low confidence'
    purely because it's cross-league, even for two clubs with decades of
    history — real Champions League backtest showed 144/144 predictions
    flagged low. An established club's overall history (any competition)
    should count, not just its (nonexistent) shared domestic-league pool."""
    league_a = Competition(slug="premier-league", name="EPL", country="England", type="league", fd_code="E0")
    league_b = Competition(slug="la-liga", name="La Liga", country="Spain", type="league", fd_code="SP1")
    ucl = Competition(slug="champions-league", name="UCL", country="Europe", type="uefa_cup")
    session.add_all([league_a, league_b, ucl])
    session.flush()

    # Two separate, properly-fittable 6-team domestic leagues (reusing the
    # same synthetic generator used elsewhere in this file), so each team
    # arrives with real domestic history but no shared Dixon-Coles pool
    # between them — the actual UEFA cross-league situation.
    teams_a = {i: Team(canonical_name=f"League A Team {i}") for i in range(6)}
    teams_b = {i: Team(canonical_name=f"League B Team {i}") for i in range(6)}
    session.add_all([*teams_a.values(), *teams_b.values()])
    session.flush()

    as_of = dt.datetime(2026, 6, 1)
    for comp, teams in ((league_a, teams_a), (league_b, teams_b)):
        rows = _synthetic_league_matches(as_of=as_of)
        session.add_all(
            [
                Match(competition_id=comp.id, utc_kickoff=r["utc_kickoff"], status="finished", home_team_id=teams[r["home_team_id"]].id, away_team_id=teams[r["away_team_id"]].id, home_goals=r["home_goals"], away_goals=r["away_goals"], source="test")
                for r in rows
            ]
        )
    session.commit()

    fixture = Match(competition_id=ucl.id, utc_kickoff=dt.datetime(2026, 6, 2), status="scheduled", home_team_id=teams_a[0].id, away_team_id=teams_b[0].id, source="test")
    session.add(fixture)
    session.commit()

    predictor = predict.Predictor(session, as_of=dt.datetime(2026, 6, 2))
    summary = predictor.predict_match(fixture)
    assert summary["confidence"] == "normal"


def test_shots_data_blends_into_established_team_prediction(session):
    """Shots-on-target should nudge the prediction relative to a goals-only
    fit when the two signals disagree, proving the blend actually executes
    rather than silently no-op'ing when shot data is present."""
    comp = Competition(slug="premier-league", name="EPL", country="England", type="league", fd_code="E0")
    session.add(comp)
    session.flush()
    teams = {i: Team(canonical_name=f"Team {i}") for i in range(6)}
    session.add_all(teams.values())
    session.flush()

    rng = np.random.default_rng(7)
    matches = []
    match_index = 0
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            for _ in range(3):
                # Team 0 dominates on shots-on-target far more than its actual
                # goals tally shows (finishing worse than its chances) —
                # shots-blend should therefore push its predicted lambda up
                # relative to a goals-only fit.
                hst = int(rng.poisson(6.0 if i == 0 else 3.0))
                ast = int(rng.poisson(3.0 if j == 0 else 3.0))
                match_index += 1
                matches.append(
                    Match(
                        competition_id=comp.id,
                        utc_kickoff=dt.datetime(2026, 1, 1) - dt.timedelta(hours=match_index),
                        status="finished",
                        home_team_id=teams[i].id,
                        away_team_id=teams[j].id,
                        home_goals=int(rng.poisson(1.2)),
                        away_goals=int(rng.poisson(1.2)),
                        home_shots_on_target=hst,
                        away_shots_on_target=ast,
                        source="test",
                    )
                )
    session.add_all(matches)
    fixture = Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 6, 2), status="scheduled", home_team_id=teams[0].id, away_team_id=teams[1].id, source="test")
    session.add(fixture)
    session.commit()

    predictor = predict.Predictor(session, as_of=dt.datetime(2026, 6, 2))
    fit = predictor.dc_fits[comp.id]
    dc_result = fit.lambdas(teams[0].id, teams[1].id)
    goals_only_home = dc_result[0]

    blended_home, _ = predictor._blend_in_shots(fixture, *dc_result)
    assert blended_home != pytest.approx(goals_only_home)
    assert comp.id in predictor.shots_fits
    assert comp.id in predictor.conversion_rates


def test_predictor_falls_back_to_elo_for_cross_league_match(session):
    comp = Competition(slug="champions-league", name="UCL", country="Europe", type="uefa_cup")
    session.add(comp)
    session.flush()
    t1, t2 = Team(canonical_name="X"), Team(canonical_name="Y")
    session.add_all([t1, t2])
    session.flush()

    fixture = Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 6, 2), status="scheduled", home_team_id=t1.id, away_team_id=t2.id, source="test")
    session.add(fixture)
    session.commit()

    predictor = predict.Predictor(session, as_of=dt.datetime(2026, 6, 2))
    lh, la, rho, weight = predictor.lambdas_for(fixture)
    assert weight == 0.0
    assert lh > 0 and la > 0


def test_predictor_confidence_low_for_thin_history(session):
    comp = Competition(slug="premier-league", name="EPL", country="England", type="league", fd_code="E0")
    session.add(comp)
    session.flush()
    rows = _synthetic_league_matches(as_of=dt.datetime(2026, 6, 1))
    teams = {i: Team(canonical_name=f"Team {i}") for i in range(6)}
    session.add_all(teams.values())
    session.flush()
    matches = [
        Match(competition_id=comp.id, utc_kickoff=r["utc_kickoff"], status="finished", home_team_id=teams[r["home_team_id"]].id, away_team_id=teams[r["away_team_id"]].id, home_goals=r["home_goals"], away_goals=r["away_goals"], source="test")
        for r in rows
    ]
    session.add_all(matches)
    session.commit()

    newly_promoted = Team(canonical_name="Newcomer FC")
    session.add(newly_promoted)
    session.flush()
    # only 2 matches for the newcomer — should be well below the shrinkage threshold
    session.add_all(
        [
            Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 5, 1), status="finished", home_team_id=newly_promoted.id, away_team_id=teams[0].id, home_goals=0, away_goals=1, source="test"),
            Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 5, 8), status="finished", home_team_id=teams[1].id, away_team_id=newly_promoted.id, home_goals=2, away_goals=0, source="test"),
        ]
    )
    fixture = Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 6, 2), status="scheduled", home_team_id=newly_promoted.id, away_team_id=teams[0].id, source="test")
    session.add(fixture)
    session.commit()

    predictor = predict.Predictor(session, as_of=dt.datetime(2026, 6, 2))
    summary = predictor.predict_match(fixture)
    assert summary["confidence"] == "low"
