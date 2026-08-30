import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Competition, Match, Team
from model import calibration, dixon_coles, elo, league_strength, ordinal, predict


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _no_network_clubelo(monkeypatch):
    """elo.compute_ratings calls out to ingest/clubelo.py's fetch_snapshot,
    which hits a real (free, unauthenticated) network API. Tests must never
    depend on that being reachable — patch it to "ClubElo has nothing for
    these teams" by default, which reproduces pre-anchor behaviour exactly
    (see _anchor_to_clubelo: an empty clubelo_ratings dict is a no-op).
    Tests that specifically exercise the anchoring logic override this
    per-test with their own monkeypatch.setattr call."""
    monkeypatch.setattr(elo.clubelo, "fetch_snapshot", lambda session, on_date: {})


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


# ---------- elo.py: ClubElo cross-league anchoring ----------


def test_clubelo_anchor_inverts_a_wrong_cross_league_ranking(session, monkeypatch):
    """Regression for the headline bug: two closed domestic pools each drift
    to ~BASE_RATING internally regardless of true strength, so a team that
    dominates a weak league can outrank one that merely holds its own in a
    strong league. ClubElo's real Man-City-vs-Porto gap (~1971 vs ~1806) is a
    live example — reproduced synthetically here so the test has no network
    dependency."""
    strong_league = Competition(slug="strong-league", name="Strong", country="X", type="league", fd_code="S0")
    weak_league = Competition(slug="weak-league", name="Weak", country="Y", type="league", fd_code="W0")
    session.add_all([strong_league, weak_league])
    session.flush()

    # "City" merely goes .500 in a league of genuine peers (internal rating
    # stays near BASE_RATING); "Porto" dominates a league of pushovers
    # (internal rating climbs well above BASE_RATING) — the internal-only
    # picture has Porto ranked above City, which is the bug.
    city = Team(canonical_name="City")
    porto = Team(canonical_name="Porto")
    session.add_all([city, porto])
    session.flush()
    city_id, porto_id = city.id, porto.id

    rng_matches = []
    as_of = dt.datetime(2026, 6, 1)
    strong_peers = [Team(canonical_name=f"Strong Peer {i}") for i in range(5)]
    weak_peers = [Team(canonical_name=f"Weak Peer {i}") for i in range(5)]
    session.add_all(strong_peers + weak_peers)
    session.flush()

    day = 0
    for peer in strong_peers:
        for home, away, hg, ag in ((city, peer, 1, 1), (peer, city, 1, 1)):
            day += 1
            rng_matches.append(Match(competition_id=strong_league.id, utc_kickoff=as_of - dt.timedelta(days=day), status="finished", home_team_id=home.id, away_team_id=away.id, home_goals=hg, away_goals=ag, source="test"))
    for peer in weak_peers:
        for home, away, hg, ag in ((porto, peer, 4, 0), (peer, porto, 0, 4)):
            day += 1
            rng_matches.append(Match(competition_id=weak_league.id, utc_kickoff=as_of - dt.timedelta(days=day), status="finished", home_team_id=home.id, away_team_id=away.id, home_goals=hg, away_goals=ag, source="test"))
    session.add_all(rng_matches)
    session.commit()

    internal = elo.compute_ratings(session, as_of=as_of)
    assert internal[porto_id] > internal[city_id]  # confirms the bug is reproduced pre-anchor

    def fake_snapshot(session, on_date):
        return {city_id: 1970.0, porto_id: 1806.0}

    monkeypatch.setattr(elo.clubelo, "fetch_snapshot", fake_snapshot)
    anchored = elo.compute_ratings(session, as_of=as_of)
    assert anchored[city_id] > anchored[porto_id]
    # Teams ClubElo covers should use its rating outright, not a blend.
    assert anchored[city_id] == pytest.approx(1970.0)
    assert anchored[porto_id] == pytest.approx(1806.0)


def test_clubelo_anchor_rescales_uncovered_teams_via_league_peers(session, monkeypatch):
    """A team ClubElo doesn't rate (e.g. a lesser Azerbaijani club) should
    still land in a sensible place on the ClubElo scale, inferred from
    peers in its own domestic league that ARE covered — not left on the raw
    internal scale, and not silently dropped."""
    league = Competition(slug="league", name="League", country="Z", type="league", fd_code="Z0")
    session.add(league)
    session.flush()

    covered = [Team(canonical_name=f"Covered {i}") for i in range(3)]
    uncovered = Team(canonical_name="Uncovered Minnow")
    session.add_all(covered + [uncovered])
    session.flush()
    uncovered_id = uncovered.id

    as_of = dt.datetime(2026, 6, 1)
    matches = []
    day = 0
    # Uncovered team is clearly the weakest of the four by internal results.
    for peer in covered:
        day += 1
        matches.append(Match(competition_id=league.id, utc_kickoff=as_of - dt.timedelta(days=day), status="finished", home_team_id=peer.id, away_team_id=uncovered.id, home_goals=3, away_goals=0, source="test"))
    # Peers also play each other so their internal ratings have real spread.
    for i, a in enumerate(covered):
        for b in covered[i + 1 :]:
            day += 1
            matches.append(Match(competition_id=league.id, utc_kickoff=as_of - dt.timedelta(days=day), status="finished", home_team_id=a.id, away_team_id=b.id, home_goals=1, away_goals=1, source="test"))
    session.add_all(matches)
    session.commit()

    internal = elo.compute_ratings(session, as_of=as_of)
    covered_ids = [t.id for t in covered]

    def fake_snapshot(session, on_date):
        # Peers rated on a scale far removed from BASE_RATING, to prove the
        # uncovered team gets rescaled rather than left near 1500.
        return {tid: 1200.0 + internal[tid] - internal[covered_ids[0]] for tid in covered_ids}

    monkeypatch.setattr(elo.clubelo, "fetch_snapshot", fake_snapshot)
    anchored = elo.compute_ratings(session, as_of=as_of)

    # Rescaled onto the ~1200 peer scale, not left near BASE_RATING (~1500).
    assert anchored[uncovered_id] < 1300
    # Still ranked below every covered peer, preserving real relative strength.
    assert all(anchored[uncovered_id] < anchored[tid] for tid in covered_ids)


def test_clubelo_anchor_noop_when_unreachable(session):
    """fetch_snapshot returning {} (network failure, see ingest/clubelo.py's
    fail-soft design) must leave ratings exactly as the pre-anchor internal
    computation produced them — the autouse fixture already patches this to
    {} for every test in this file, so this just asserts that contract."""
    t1, t2 = Team(canonical_name="A"), Team(canonical_name="B")
    session.add_all([t1, t2])
    session.flush()
    session.add(Match(competition_id=1, utc_kickoff=dt.datetime(2026, 1, 1), status="finished", home_team_id=t1.id, away_team_id=t2.id, home_goals=2, away_goals=0, source="test"))
    session.commit()

    ratings = elo.compute_ratings(session)
    assert ratings[t1.id] > elo.BASE_RATING
    assert ratings[t2.id] < elo.BASE_RATING


# ---------- predict.py: rest-days / fixture-congestion adjustment ----------


def test_rest_adjustment_penalises_short_rest(session):
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
    # Team 0's most recent match is only 2 days before the fixture below —
    # fixture congestion (e.g. a midweek European tie).
    session.add(Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 5, 30), status="finished", home_team_id=teams[0].id, away_team_id=teams[2].id, home_goals=1, away_goals=1, source="test"))
    session.commit()

    fixture = Match(competition_id=comp.id, utc_kickoff=dt.datetime(2026, 6, 1), status="scheduled", home_team_id=teams[0].id, away_team_id=teams[1].id, source="test")
    session.add(fixture)
    session.commit()

    predictor = predict.Predictor(session, as_of=dt.datetime(2026, 6, 1))
    lambda_home, lambda_away, rho, weight = predictor.lambdas_for(fixture)
    adjusted_home, adjusted_away = predictor._apply_rest_adjustment(fixture, lambda_home, lambda_away)

    assert adjusted_home < lambda_home  # short-rested home team penalised
    assert adjusted_away == pytest.approx(lambda_away)  # away team unaffected (no recent match on record)


def test_rest_adjustment_noop_for_well_rested_team(session):
    t1, t2 = Team(canonical_name="A"), Team(canonical_name="B")
    session.add_all([t1, t2])
    session.flush()
    session.add(Match(competition_id=1, utc_kickoff=dt.datetime(2026, 5, 1), status="finished", home_team_id=t1.id, away_team_id=t2.id, home_goals=1, away_goals=1, source="test"))
    session.commit()

    fixture = Match(competition_id=1, utc_kickoff=dt.datetime(2026, 6, 1), status="scheduled", home_team_id=t1.id, away_team_id=t2.id, source="test")
    session.add(fixture)
    session.commit()

    predictor = predict.Predictor(session, as_of=dt.datetime(2026, 6, 1))
    adjusted_home, adjusted_away = predictor._apply_rest_adjustment(fixture, 1.5, 1.2)
    assert adjusted_home == pytest.approx(1.5)  # a month's rest -> no penalty
    assert adjusted_away == pytest.approx(1.2)


def test_rest_adjustment_noop_for_team_with_no_prior_match(session):
    t1, t2 = Team(canonical_name="Debutant"), Team(canonical_name="Established")
    session.add_all([t1, t2])
    session.flush()

    fixture = Match(competition_id=1, utc_kickoff=dt.datetime(2026, 6, 1), status="scheduled", home_team_id=t1.id, away_team_id=t2.id, source="test")
    session.add(fixture)
    session.commit()

    predictor = predict.Predictor(session, as_of=dt.datetime(2026, 6, 1))
    adjusted_home, adjusted_away = predictor._apply_rest_adjustment(fixture, 1.5, 1.2)
    assert adjusted_home == pytest.approx(1.5)
    assert adjusted_away == pytest.approx(1.2)


# ---------- calibration.py ----------


def test_temperature_scaling_is_noop_at_one():
    probs = (0.6, 0.25, 0.15)
    assert calibration.apply_temperature(probs, 1.0) == probs


def test_fit_temperature_corrects_overconfident_model():
    """Model always screams 90% confidence in the actual winner but is only
    right 50% of the time — classic overconfidence. Fitting should find T > 1
    (pulling predictions toward uniform) and that correction should reduce
    mean log-loss on the same records relative to leaving T=1."""
    rng = np.random.default_rng(3)
    records = []
    for _ in range(500):
        actual = int(rng.integers(0, 3))
        if rng.random() < 0.5:
            probs = [0.05, 0.05, 0.05]
            probs[actual] = 0.9
        else:
            # Wrong half the time: confidently picks a *different* outcome.
            wrong = (actual + 1) % 3
            probs = [0.05, 0.05, 0.05]
            probs[wrong] = 0.9
        records.append({"probs": tuple(probs), "actual": actual})

    T = calibration.fit_temperature(records)
    assert T > 1.0

    baseline_loss = calibration._mean_log_loss(1.0, records)
    corrected_loss = calibration._mean_log_loss(T, records)
    assert corrected_loss < baseline_loss


def test_fit_temperature_empty_records_is_noop():
    assert calibration.fit_temperature([]) == 1.0


def test_apply_temperature_preserves_probability_simplex():
    q = calibration.apply_temperature((0.7, 0.2, 0.1), 2.5)
    assert sum(q) == pytest.approx(1.0)
    assert all(0.0 <= x <= 1.0 for x in q)


# ---------- ordinal.py ----------


def _synthetic_ordinal_rows(n=400, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        elo_diff = float(rng.normal(0, 150))
        home_advantage = 1.0
        rest_diff = float(rng.normal(0, 2))
        # True latent process: home team favoured by elo_diff, home advantage,
        # and more rest; noisy logistic outcome.
        eta = elo_diff / 100 + 0.4 * home_advantage + 0.1 * rest_diff + rng.logistic(0, 1)
        if eta > 0.6:
            outcome = 0  # home
        elif eta > -0.6:
            outcome = 1  # draw
        else:
            outcome = 2  # away
        rows.append({"elo_diff": elo_diff, "home_advantage": home_advantage, "rest_diff": rest_diff, "outcome": outcome})
    return rows


def test_ordinal_fit_returns_none_below_minimum_rows():
    assert ordinal.fit([{"elo_diff": 0, "home_advantage": 1, "rest_diff": 0, "outcome": 0}] * 10) is None


def test_ordinal_fit_recovers_positive_elo_effect():
    rows = _synthetic_ordinal_rows()
    fit = ordinal.fit(rows)
    assert fit is not None
    assert fit.beta[0] > 0  # higher elo_diff -> more home-favoured


def test_ordinal_predict_proba_sums_to_one_and_is_monotonic():
    rows = _synthetic_ordinal_rows()
    fit = ordinal.fit(rows)
    weak_home = fit.predict_proba(elo_diff=-200, home_advantage=1, rest_diff=0)
    strong_home = fit.predict_proba(elo_diff=200, home_advantage=1, rest_diff=0)
    assert sum(weak_home) == pytest.approx(1.0, abs=1e-6)
    assert sum(strong_home) == pytest.approx(1.0, abs=1e-6)
    assert strong_home[0] > weak_home[0]  # home-win prob rises with elo_diff
    assert strong_home[2] < weak_home[2]  # away-win prob falls
