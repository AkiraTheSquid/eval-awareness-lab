# 04 · How the two agents work together

Two Claude Code sessions share this repository: the **Windows agent** (runs experiments on the
brother's PC) and the **Linux agent** (on Seth's PC; reviews results, turns them into Delta Drills
content). Neither can see the other's chat. Everything the other side needs must be in the repo.

## Who owns what

| path | owner | the other side |
|---|---|---|
| `lab/`, `runs/`, `exercises/`, `STATUS.md` | Windows agent | reads; asks via issues |
| `docs/`, `reference/`, `AGENTS.md`, `README.md` | Linux agent | reads; asks via issues |

Push directly to `main` for files you own (`git pull --rebase` first, small commits, clear messages).
To change something the other side owns, open an issue or a pull request and let them merge it.

## Channels

- **GitHub issues are the task board.** One issue per experiment (`E0` … `E13`) and one for setup.
  Labels: `ready` (can start), `running`, `needs-review` (results pushed, Linux agent to check),
  `needs-seth` (a decision only Seth can make), `blocked`.
  - When you start an issue: comment with your plan and size estimate, set `running`.
  - When results are pushed: comment with the commit sha, the headline numbers with intervals, and
    whether every §6 check passed; set `needs-review`.
  - The Linux agent reviews, comments, and either closes the issue or reopens it with what to fix.
- **`STATUS.md` is the live dashboard** (Windows agent owns it). Keep it short: environment, speed
  table, current run (what, started when, expected end), predictions written before each run, and the
  latest result per experiment. Overwrite stale lines rather than appending forever.
- **Questions for Seth** go in the chat with him *and* in an issue labelled `needs-seth`, so the
  Linux agent sees the decision too.

## Commands

```bash
gh issue list --label ready
gh issue comment <n> --body-file comment.md
gh issue edit <n> --add-label running --remove-label ready
gh issue create --title "..." --body-file body.md --label needs-seth
```

## Checking for the other side's work

At the start of every session and before every push: `git pull --rebase` and
`gh issue list --state open`. Reply to anything addressed to you before starting new work.
