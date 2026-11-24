"""Generate synthetic online-casino data: a players table and a player-day activity table.

Each player moves through three latent states: engaged -> fading -> gone, with
occasional win-backs. The daily hazard of starting to fade depends on static
traits (channel, game, age, latent engagement), tenure (new players leave more
often), recent luck (losing streaks push players out) and recent bonuses (which
retain them). The latent state itself is never exported, so the models have to
infer it from behaviour.

Run with ``python -m churn.generate_data``.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from churn import config

logger = logging.getLogger(__name__)

# Per game: (bets per session, house edge, per-bet outcome volatility, stake multiplier).
GAME_PROFILE = {
    "slots": (60, 0.040, 4.0, 0.5),
    "live_casino": (25, 0.020, 1.2, 3.0),
    "table_games": (30, 0.015, 1.0, 2.0),
    "crash_games": (40, 0.030, 3.0, 1.0),
}
CHANNEL_SHARE = {"organic": 0.35, "affiliate": 0.30, "paid_ads": 0.25, "referral": 0.10}
GAME_SHARE = {"slots": 0.55, "live_casino": 0.20, "table_games": 0.15, "crash_games": 0.10}

# Log-hazard effects on the engaged -> fading transition.
CHANNEL_HAZARD = {"organic": -0.20, "affiliate": 0.30, "paid_ads": 0.45, "referral": -0.30}
GAME_HAZARD = {"slots": 0.0, "live_casino": -0.25, "table_games": -0.10, "crash_games": 0.30}
BASE_FADE_HAZARD = 0.007  # per day, for an average established player
FADING_TO_GONE = 1 / 12  # per day
FADING_TO_ENGAGED = 0.02  # per day, spontaneous win-back
GONE_TO_ENGAGED = 0.001  # per day, reactivation long after leaving

NOT_REGISTERED, ENGAGED, FADING, GONE = 0, 1, 2, 3


def generate(
    n_players: int = config.N_PLAYERS,
    start: pd.Timestamp = config.DATA_START,
    end: pd.Timestamp = config.DATA_END,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate players and their daily activity.

    Parameters
    ----------
    n_players : int
        Number of players to simulate.
    start, end : pd.Timestamp
        First and last simulated calendar day (both inclusive).
    seed : int
        Seed of the random generator.

    Returns
    -------
    players : pd.DataFrame
        One row per player: ``player_id``, ``registration_date``, ``age``,
        ``acquisition_channel``, ``favorite_game``.
    activity : pd.DataFrame
        One row per player and active day: ``player_id``, ``date``,
        ``sessions``, ``bets``, ``turnover``, ``ggr``, ``deposits``,
        ``withdrawals``, ``bonus``.

    Raises
    ------
    ValueError
        If ``n_players`` is not positive or ``end`` precedes ``start``.
    """
    if n_players <= 0:
        raise ValueError(f"n_players must be positive, got {n_players}")
    n_days = (end - start).days + 1
    if n_days <= 0:
        raise ValueError(f"end ({end.date()}) precedes start ({start.date()})")

    rng = np.random.default_rng(seed)
    channels, games = list(CHANNEL_SHARE), list(GAME_SHARE)

    # Static traits.
    reg_day = rng.integers(0, n_days, n_players)
    age = np.clip(18 + rng.gamma(4.0, 5.0, n_players), 18, 80).astype(int)
    channel = rng.choice(channels, n_players, p=list(CHANNEL_SHARE.values()))
    game = rng.choice(games, n_players, p=list(GAME_SHARE.values()))
    engagement = rng.standard_normal(n_players)  # latent, not exported
    bonus_affinity = rng.beta(2, 5, n_players)  # latent, not exported

    profile = np.array([GAME_PROFILE[g] for g in game])
    bets_per_session, edge, volatility, stake_mult = profile.T
    stake_level = np.exp(rng.normal(np.log(1.5), 0.8, n_players)) * stake_mult
    session_rate = 0.5 + 0.8 * expit(engagement)
    # Scale of a typical day's net result, used to standardise luck.
    typical_bets = (1 + session_rate) * bets_per_session
    luck_scale = stake_level * volatility * np.sqrt(typical_bets)

    activity_logit = -0.4 + engagement + np.where(game == "crash_games", 0.2, 0.0)
    static_log_hazard = (
        np.log(BASE_FADE_HAZARD)
        - 0.6 * engagement
        - 0.012 * (age - 40)
        + np.array([CHANNEL_HAZARD[c] for c in channel])
        + np.array([GAME_HAZARD[g] for g in game])
    )

    state = np.full(n_players, NOT_REGISTERED)
    luck_ewma = np.zeros(n_players)  # > 0 means the player has been losing
    bonus_ewma = np.zeros(n_players)
    chunks: list[dict[str, np.ndarray]] = []

    for day in range(n_days):
        new = reg_day == day
        state[new] = ENGAGED
        tenure = day - reg_day
        fading = state == FADING
        weekend = (start + pd.Timedelta(days=day)).day_of_week >= 4  # Fri-Sun

        # Who plays today: honeymoon boost for new players, a drop once fading.
        p_active = expit(activity_logit + np.exp(-tenure / 14) + 0.3 * weekend)
        p_active = np.where(fading, 0.3 * p_active, p_active)
        active = ((state == ENGAGED) | fading) & (rng.random(n_players) < p_active)
        active |= new  # registration always comes with a first session
        idx = np.flatnonzero(active)
        m = idx.size

        sessions = 1 + rng.poisson(session_rate[idx])
        bets = 1 + rng.poisson(sessions * bets_per_session[idx])
        stake = stake_level[idx] * rng.lognormal(0.0, 0.25, m) * np.where(fading[idx], 0.6, 1.0)
        turnover = bets * stake
        # House win: the edge plus zero-mean luck that averages out over bets.
        ggr = turnover * (edge[idx] + volatility[idx] * rng.standard_normal(m) / np.sqrt(bets))
        lost = ggr > 0
        deposit_today = new[idx] | (rng.random(m) < 0.15 + 0.35 * lost)
        deposits = np.where(
            deposit_today,
            (20 * stake_level[idx] + np.maximum(ggr, 0)) * rng.lognormal(0.0, 0.4, m),
            0.0,
        )
        withdrawals = np.where(~lost & (rng.random(m) < 0.35), -ggr * rng.uniform(0.4, 0.9, m), 0.0)
        got_bonus = rng.random(m) < 0.04 + 0.25 * bonus_affinity[idx]
        bonus = np.where(got_bonus, 10 * stake_level[idx] * rng.lognormal(0.0, 0.3, m), 0.0)

        chunks.append(
            {
                "player_idx": idx,
                "day": np.full(m, day),
                "sessions": sessions,
                "bets": bets,
                "turnover": turnover,
                "ggr": ggr,
                "deposits": deposits,
                "withdrawals": withdrawals,
                "bonus": bonus,
            }
        )

        # Behavioural memory that feeds tomorrow's hazard.
        luck_ewma[idx] = 0.85 * luck_ewma[idx] + 0.15 * ggr / luck_scale[idx]
        bonus_ewma *= 0.93
        bonus_ewma[idx] += got_bonus

        log_hazard = (
            static_log_hazard
            + np.log1p(3 * np.exp(-tenure / 20))
            + 0.3 * np.clip(luck_ewma / 0.3, -2, 2)
            - 0.35 * np.minimum(bonus_ewma, 2)
        )
        u = rng.random(n_players)
        start_fading = (state == ENGAGED) & (u < np.exp(log_hazard))
        leave = fading & (u < FADING_TO_GONE)
        win_back = fading & (u > 1 - FADING_TO_ENGAGED)
        reactivate = (state == GONE) & (u < GONE_TO_ENGAGED)
        state[start_fading] = FADING
        state[leave] = GONE
        state[win_back | reactivate] = ENGAGED

    columns = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    money = ("turnover", "ggr", "deposits", "withdrawals", "bonus")
    activity = pd.DataFrame(
        {
            "player_id": columns["player_idx"] + 1,
            "date": start + pd.to_timedelta(columns["day"], unit="D"),
            "sessions": columns["sessions"],
            "bets": columns["bets"],
            **{name: columns[name].round(2) for name in money},
        }
    )
    players = pd.DataFrame(
        {
            "player_id": np.arange(1, n_players + 1),
            "registration_date": start + pd.to_timedelta(reg_day, unit="D"),
            "age": age,
            "acquisition_channel": channel,
            "favorite_game": game,
        }
    )
    return players, activity


def main() -> None:
    """Command-line entry point: simulate and write the two parquet files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-players", type=int, default=config.N_PLAYERS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out-dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    players, activity = generate(n_players=args.n_players, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    players.to_parquet(args.out_dir / "players.parquet", index=False)
    activity.to_parquet(args.out_dir / "activity.parquet", index=False)
    logger.info(
        "Wrote %d players and %d player-day rows (%s .. %s) to %s",
        len(players),
        len(activity),
        activity["date"].min().date(),
        activity["date"].max().date(),
        args.out_dir,
    )


if __name__ == "__main__":
    main()
