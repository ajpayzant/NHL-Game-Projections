# How the projections work

This tool projects, for one specific upcoming game, every dressed skater's **ice time, shots on
goal, goals, assists and points**, and the chance of hitting common lines (1+ point, a goal,
2+/3+/4+ shots...). This page explains what it is built from, how a projection is put
together, what was tested, and how far to trust it.

---

## 1. The idea in one paragraph

A player's stat line in a game is **how much he plays** × **how productive he is per minute**
× **who he is playing against tonight**. The model estimates each of those three things
separately, from his history and tonight's line-up, and multiplies them. Every number on the
Player page's "How tonight's projection is calculated" panel is one of those pieces, so any
projection can be traced back to its inputs.

---

## 2. The data

Everything comes from the free NHL API (2018-19 to today, ~380,000 skater-games) plus Daily
Faceoff for tonight's lines:

| Source | What we take from it |
|---|---|
| Play-by-play | every shot, goal (with both assisters), penalty and faceoff, with the game situation (5v5, PP, PK, empty net) |
| Shift charts | who was on the ice every second, so we know each player's linemates, his on-ice goals for, and his real line / pair / PP unit in every past game |
| Official time-on-ice report | each player's EV, PP and PK minutes per game |
| Box scores, player pages | who dressed; age, size, draft position |
| Daily Faceoff | tonight's lines, PP / PK units, injuries, starting goalies (logged every time they are read) |

PuckPedia blocks automated access and MoneyPuck requires a licence, so neither is used;
everything they offer for this purpose can be rebuilt from the shift charts.

---

## 3. How a projection is built, step by step

### Step 1: Ice time (the biggest driver)

Ice time is split into even strength (EV), power play (PP) and penalty kill (PK), because a
player's production per minute is very different in each.

1. **Team minutes tonight.** How many PP minutes the team should get depends on how often it
   draws penalties and how often tonight's opponent takes them (and vice versa for PK). The
   rest of the game is EV. Playoff games get extra (20-minute overtimes).
2. **The player's share** of those minutes. Two things decide it:
   - his own recent share (weighted toward the last ~4 games, with a 16-game version for stability);
   - what **his team usually gives the slot he is in tonight** (first-line forwards typically
     get 11-12% of a team's forward EV minutes, PP1 players 60-80% of its PP time, and each
     team's own pattern is used).
   How much weight the slot gets depends on how fresh the line information is: a Daily Faceoff
   chart still marked "Last Game" counts like last game's lines; one updated by a beat reporter
   for tonight, or a manual override, counts more.
3. **Everything adds up.** Shares are rescaled so the team's forwards fill exactly the forward EV
   minutes, five skaters fill each PP minute and four fill each PK minute. Scratch a player and
   his minutes go to his teammates; move someone to the first line and someone else loses time.

### Step 2: Per-minute ratings (how productive he is)

For each player we keep a small set of ratings, each measured **per minute, by situation**:

| Rating | Memory (half-life) | Pulled toward a baseline by |
|---|---|---|
| Shots on goal per minute (EV, PP, PK) | ~80 games | 120 EV minutes |
| Shooting % | ~320 games (about four seasons) | 200 shots |
| Team goals per minute while he is on the ice (EV, PP) | ~80 games | 120 EV minutes |
| His share of those goals as assists | ~160 games | 20 on-ice goals |

Three things make these ratings fair to every kind of player:

- **Recency weighting.** Recent games count more, smoothly (a game 80 games ago counts half as
  much as last night). Summers count as extra time passing.
- **Regression toward a role baseline, not the league average.** A player with little history is
  pulled toward what a player *in his role* (ice-time share, PP / PK share, position, NHL
  experience) typically does, not toward the average skater. A fourth-line call-up starts from a
  fourth-liner's baseline; a first-line PP1 player starts from a first-liner's. Without this,
  depth players and rookies were inflated by about 17% and stars were held back. Players with a
  lot of history are barely affected; their own record dominates.
- **Relative to the league at the time.** League-wide scoring moves (shots per minute fell about
  12% from 2021 to 2025 while shooting % rose). Each past game is compared with what an average
  player would have done *that season*, then the rating is applied to the league as it is now,
  so old seasons don't drag projections up or down.

**How much regression, in practice.** The weight a player's own history gets going into
2026-27 (the rest goes to his role baseline), median by group:

| | Shots/min EV | Shots/min PP | Shooting % | On-ice goals EV | Assist share |
|---|---|---|---|---|---|
| Stars (0.9+ pts/gm) | 92% | 87% | 79% | 91% | 88% |
| Top-6 / top-4 (0.5-0.9) | 91% | 82% | 72% | 91% | 85% |
| Regulars | 90% | 38% | 60% | 89% | 80% |
| Young / new (<120 games) | 77% | 14% | 20% | 77% | 48% |

Stars are barely regressed on shot volume and on-ice scoring; shooting % is regressed the most,
because one season of shooting is mostly luck. Regression was tested at 75% and 50% of this
strength (walk-forward, 2023-26): every stat got worse (points deviance 0.9036 → 0.9039 → 0.9044),
depth players became inflated, and stars were *not* helped (actual ÷ projected 1.013 → 1.016 →
1.021). The current strength is the best tested.

### Step 3: From ratings to tonight's numbers

- **Shots** = Σ (minutes in each situation × shots per minute) × context
- **Goals** = projected shots × his shooting % × context
- **Assists** = Σ (minutes × team goals per minute while he's on × his assist share) × context
- **Points** = goals + assists
- **Power-play points** = PP minutes × team PP goals per minute while he's on × his share of
  those goals as a scorer or assister (his PP "IPP", ~160-game memory), × the opponent's penalty
  kill (PP goals it allows vs league), opposing goalie, home ice and age. Walk-forward it beats the
  season-to-date average by 13% (deviance 0.351 vs 0.405), as much as the other stats, and its
  P(1+ PPP) is calibrated within 1-2 points.

### Step 4: Context, tonight's opponent and situation

Small multipliers, each fitted from the data (1.00 = no effect):

| Context | Shots | Goals | Assists |
|---|---|---|---|
| Opponent's shot / goal suppression (vs league) | power ≈0.92 | (via shots) | power ≈0.78 |
| Opposing goalie (goals saved above expected) | | −8% per +1% save rate | −2.5% |
| Home ice | +4% | +2% | +7% |
| Rink scorer (some arenas count shots generously) | power ≈0.3 | | |
| Back-to-back (own / opponent's) | −3% / +2% | | |
| Linemates better / worse than his usual ones | | | power ≈0.10 |
| Age: per year younger than 28 / older than 28 | +0.5% / −0.3% | +0.1% / −0.6% | +0.3% / −0.9% |

The age terms correct a lag: a multi-year history trails a young player who is still improving
and flatters a veteran who is declining.

### Step 5: Probabilities

Goals and assists follow a Poisson distribution around their projections; shots are slightly
more spread out than Poisson (negative binomial, α≈0.05). That gives P(1+ point), P(goal),
P(3+ SOG) and so on. These probabilities were checked against what actually happened (section 5).

**Fair odds.** Any probability converts to a no-vig American price: 60% → −150, 25% → +300,
50% → EVEN. "SOG o2.5" is the same event as 3+ SOG, "Points o0.5" is 1+ point. A sportsbook's
price includes its margin, so a bet only has value when the book pays more than the fair price.

### Step 6: Team markets (moneyline, puck line, total)

A team's expected goals is the sum of its skaters' projected goals. The final score is then
built the way hockey actually ends:

1. **Regulation goals** for each team (Poisson), from expected goals less the part scored in
   overtime or into empty nets.
2. **Tied games are tighter** than independent scoring implies: ties after regulation are made
   more likely (fitted: +77%), matching the ~22% of games that reach OT.
3. **Empty nets**: a team leading late adds an empty-net goal with probability 49% (1-goal lead),
   65% (2-goal) or 18% (3-goal). This is why 2-goal wins are rarer than 3-goal wins in the NHL,
   and it matters most for the ±1.5 puck line.
4. **Overtime / shootout**: the winner gets +1 goal (as books grade it); the stronger team wins it
   slightly more often.
5. Team strength gaps are widened a little (power 1.4), because summing individually-regressed
   player projections compresses differences between teams.

These few numbers were fitted on 2023-24 and checked on 2024-25 and 2025-26 (2,792 games):
home win 54.6% projected vs 54.2% actual; favourite -1.5 37.5% vs 36.0%; over 5.5 54.7% vs
55.7%; over 6.5 43.9% vs 44.6%. A plain Poisson model got the puck line badly wrong (32% vs 36%).
The total shown is the x.5 line closest to 50%, with its fair over / under price.

---

## 4. What was tested to get here

The model was chosen by comparing alternatives on seasons the model had not seen (walk-forward:
each season is projected using only earlier seasons).

**Every data type, alone and together.** 102 pregame variables in ten groups: usage, rates,
raw averages (last 5/10/20, season, last season, career), shot attempts and shot quality (xG),
on-ice for/against, zone starts and penalties drawn, physical/style (hits, blocks, takeaways,
giveaways), bio (age, height, weight, draft position), team and opponent context, and linemates.
Each group was added on top of the model to see if it carried anything the model missed:

| Group added | Gain |
|---|---|
| Line / slot usage (for ice time) | **~2%** → built into the ice-time model |
| Raw recent averages (last 5/10/20 games) | ≤0.4% |
| Everything else (xG, attempts, zone starts, hits, size, draft, team strength, ...) | ≤0.1% each |

**Different kinds of model.** Gradient boosting on all 102 variables, boosting on simple stats
only, a wide regularised regression, boosting the model's errors, and blends of these. None beat
the structured model by more than ~0.3%, and only for assists/points, so the transparent model
was kept and the one real gap the machine-learning models found (how line slots drive ice
time) was added to it explicitly; after that it beat them on ice time too.

**Fairness to every kind of player.** Projections were checked against results by last
season's scoring tier, position (C, LW, RW, D), experience and age. Issues found and fixed:
rookies/call-ups inflated (fixed with role baselines), stars and defencemen held back (role
baselines + more history), the linemate term penalising stars (now compared with *his usual*
linemates, not with himself), age lag (age terms), playoff and top-line ice time (playoff terms
and freshness-weighted slots).

**What didn't help.** Hot/cold streaks beyond ice time (a last-5-game form term adds nothing),
shot-quality xG instead of shooting %, direct goals-per-60, opponent shot quality, the opposing
goalie beyond a small effect, zone starts, physical stats, size, draft position.

---

## 5. How accurate it is

Walk-forward over 2023-24, 2024-25 and 2025-26, regular season (141,346 player-games); lines
assumed = the player's previous game (a deliberately conservative stand-in for the morning line chart):

| | TOI miss (min) | SOG miss | Points miss | Points deviance* |
|---|---|---|---|---|
| **This model** | **1.79** | **1.024** | **0.520** | **0.904** |
| Season-to-date average | 2.00 | 1.042 | 0.522 | 0.977 |
| Last-10 average | 1.92 | 1.070 | 0.525 | 1.295 |
| This model with tonight's actual lines | 1.51 | 1.017 | 0.517 | 0.898 |

*Deviance is the proper score for count predictions; lower is better. The plain "miss" barely
separates methods for points because most games are 0 or 1 point. The probabilities are where
the gain shows.

**Calibration.** When the model says 25%, it happens about 25% of the time: across all ranges
P(1+ point), P(goal), P(1+ assist) and P(2+/3+/4+ SOG) match actual frequencies within 1-3 points
(see the chart below). Season averages are within 1% of actual.

**Fairness check** (actual ÷ projected points, 1.00 = exact):

| Group | Points |
|---|---|
| Stars (≥1.0 pts/gm last season) | 1.02 |
| 0.8-1.0 pts/gm | 1.02 |
| Middle tiers | 0.99-1.00 |
| Depth (<0.2 pts/gm) | 1.00 |
| New players (<20 NHL games) | 0.96 |
| C / LW / RW / D | 1.00 / 0.99 / 1.01 / 1.00 |

Stars are still projected about 2% low on points (mostly assists) and brand-new players about
4% high; both are much smaller than before (from 3-4% and 17%), and the remaining star gap is
inside game-to-game noise for any one player.

**How much luck there is.** A player's ice time varies about ±2.5 minutes around his own average
game to game; a projection of 1.0 point still leaves ~35% chance of zero. The model gets the
expectations and probabilities right; no pregame model can call single games.

---

## 6. Using it

- **Probabilities / Fair odds** (sidebar on Slate and Game): shows the same columns as % or as fair odds.
- **Slate**: every skater on the date, sortable; download as CSV.
- **Game**: both teams' projected goals, assists, shots, points, PP and PK minutes; a fair game-lines
  board (moneyline, puck line, total); then each team's players in line order.
- **Slate**: also has a "Game lines" panel with every game's fair moneyline, puck line and total.
- **Lines & goalies**: the depth chart (LW-C-RW, LD-RD, starter/backup) and PP / PK units, editable;
  or a table with a dressed box and a TOI override. Saving updates every projection at once.
- **Player**: tonight's projection vs his last 10 / season / career; a prop-line table (SOG o1.5-o4.5,
  points o0.5-o2.5, goals o0.5, assists o0.5-o1.5, PP points o0.5-o1.5) with the model's %, fair over / under odds and
  his hit rates over different windows and against tonight's opponent; his game log, season
  history and splits.

The line-slot weights were fitted without historical *pregame* line charts (none exist); the
tool logs every Daily Faceoff read so they can be refitted on real pregame lines after a few
weeks of the season.
