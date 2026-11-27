# Playbooks

A risk score tells you who to worry about. A playbook tells you what to do about it. Every scored account gets a play from the playbook: what to do, who leads it, what to say on the next call, and how you'll know it worked. The account's risk tier then sets how hard to run it.

## What a play contains

| Field | What it's for | Example (stalled pilot) |
|---|---|---|
| Name | What the team calls the play | Pilot-to-production push |
| Owner | The roles that lead it | CSM + Solutions Engineer |
| Actions | Three concrete steps | Agree written success criteria and a go-live date with the pilot sponsor; ship one production workflow before the pilot ends; line up the economic buyer and a core budget line |
| Talk track | What the rep says on the next call, in plain words | "The pilot has shown what's possible, and I want to make sure it turns into results your team keeps. Can we agree on what success looks like and pick one workflow to put into production this month?" |
| Exit criteria | How you know the play worked, so it doesn't run forever | A production workflow is live and the economic buyer has agreed the conversion budget |

On top of the play, the account gets a **timing window** from its days to renewal (overdue, negotiation, engagement, planning, early) and a **motion** from its risk tier:

| Tier | Motion | Cadence | Escalation |
|---|---|---|---|
| critical | Save | Review the save plan weekly with the CS leader until the renewal is signed | Executive sponsor from both companies; marked at risk in the forecast |
| high | Recover | Review progress every two weeks with the CS leader | Marked at risk in the forecast call; an executive joins if it moves to critical |
| watch | Monitor | CSM checks the signal monthly and logs what changed | Raised with the CS leader if the account moves up a tier |
| low | Standard | Normal renewal cadence | None needed |

## How a play is chosen

1. **An alert names the cause.** A high or medium alert with a play wins, highest severity first. A champion who left, a competitor evaluation or an invoice 60 days late is more specific than any model driver.
2. **Watermelons get checked.** A green account the model rates as risky, with no such alert, gets the watermelon check: ask the customer directly instead of trusting the score.
3. **Healthy accounts get a healthy renewal.** A low-tier account with no such alert gets the healthy renewal play (confirm intent and signer, send a value summary, ask about expansion). Its model drivers are mild pushes on a low probability, not a reason to run a save play.
4. **Otherwise, the top driver.** The play for the scorecard group behind the account's strongest actionable risk driver, such as adoption recovery or burn recovery. Contract size and segment describe the account rather than a cause anyone can fix, so they never pick a play.
5. **Otherwise, the default renewal motion.**

Each account also gets up to two **supporting plays**: the watermelon check first if it applies, then the plays for its other alerts, including low-severity ones such as a first renewal. They show as "Also run" in the app and in `supporting_plays` in the CSV.

## Where plays show up

- **App, Account detail:** the play with its motion, owner and timing, the steps, the cadence and escalation, the talk track, the exit criteria and any supporting plays.
- **App, Portfolio:** "Plays to run" lists each primary play across accounts above the low tier, with the accounts by tier, ARR and expected loss, so a CS leader can see where the team's time needs to go.
- **`report.md`:** the same "Plays to run" table, plus what each tier's motion means.
- **`scored_accounts.csv`:** `save_play`, `play_id`, `play_owner`, `play_timing`, `play_motion`, `play_cadence`, `play_escalation`, `play_actions`, `play_talk_track`, `play_exit_criteria` and `supporting_plays`.
- **`summary.json`:** `plays` (the table above) and `play_tiers`.
- **Account briefs:** the rules-based brief uses the play's talk track and adds "Done when" from its exit criteria. The Claude-written brief is told to build on the play, and its "Done when" is the playbook's exit criteria, word for word.

## Bring your own playbook

The built-in plays are in [`src/renewal_risk/playbooks/default.yaml`](../src/renewal_risk/playbooks/default.yaml). To change them, write a YAML file with only the entries you want to change. Anything you leave out comes from the built-in playbook, and a play you define replaces the built-in play with the same key.

```yaml
# our_playbook.yaml
alerts:
  champion_left:
    name: Champion rescue
    owner: CS leader + Account Executive
    actions:
      - Call the new decision maker within 48 hours
      - Send our one-page value summary the same day
      - Book an executive business review within two weeks
    talk_track: >-
      I heard about the change on your side and wanted to reach out right away. I'd like to show you what
      your team has achieved with us and hear what matters most to you.
    exit_criteria: New decision maker met and an executive business review booked
tiers:
  critical:
    cadence: Daily save-desk standup until the renewal is signed
```

```bash
renewal-risk playbook --playbook our_playbook.yaml       # validate it and check coverage
renewal-risk score my_renewals.csv --model model.json --playbook our_playbook.yaml
renewal-risk demo --playbook our_playbook.yaml
```

The sections are:

- `alerts`: plays keyed by alert id (see the `alerts` list in each profile under `src/renewal_risk/profiles/`).
- `groups`: plays keyed by scorecard group, used when no alert names the cause.
- `special`: the three fixed plays, `watermelon`, `healthy` and `default`.
- `tiers`: `motion`, `cadence` and `escalation` for each tier name.

Every play needs `name`, `owner`, `actions` (a list), `talk_track` and `exit_criteria`. A missing field, an unknown field or an unknown section stops with a message that says which entry to fix.

`renewal-risk playbook` checks that every alert and scorecard group in every built-in profile has a play, and exits with an error if one doesn't. Pass `--config my_profile.yaml` to check a custom profile instead. **If you add an alert**, in a custom profile or a data add-on, add a play for it too. The test suite enforces this for the built-in profiles, so an alert can't ship without a play.

## Where the plays come from

The plays follow common renewals practice rather than a published standard: fix the specific cause, put the right owner on it, multi-thread the relationship, give notice on price changes early, and pace the work to the 180, 120 and 90-day renewal cadence the scorecard's urgency curve already uses. The structure (a trigger, an owner, steps and exit criteria) mirrors the calls to action and success plans in CS platforms such as Gainsight, and the motions mirror TSIA's LAER model (land, adopt, expand, renew). Treat them as a starting point to tune to your team, and track which plays actually save renewals: the scorer ranks risk, it doesn't prove a play works.
