# LLM ego agent

`eval_llm.py` (repository root) evaluates a large language model as the ego agent in
Overcooked, paired with the same rule-based Level-0 and trained Level-1 partners as the
learning agents and with the same protocol. The modules here turn the game into text, call
the model, and execute its decisions.

| file | role |
|---|---|
| `state_text.py` | describes the kitchen as text: ASCII map with legend, recipe table, what both agents hold, plate contents, the objects the agent can reach with the cell to stand on and the direction to face; also holds the system prompts |
| `skills.py` | high-level skill set and its controller: shortest-path execution, precondition checks, completion checks |
| `memory.py` | round memory: the events of every earlier episode of the round and a summary of each |
| `policy.py` | the agent: builds the prompt, queries the model, parses the reply, falls back to no-op / `wait(1)` on an error or an unparseable reply |
| `client.py` | minimal OpenAI-compatible chat client (standard library only), with call statistics |
| `actions.py` | action constants and parsing of low-level replies |

## What the model sees and does

At every decision the model receives the rules (system prompt) and the current state as
text. The prompt contains no hint about which seat can deliver and no example strategies.

Two interfaces (`--actions`):

- **`high`** (the default of the launch scripts): the model picks one skill from a fixed set, following
  ProAgent (Zhang et al., 2024). The preconditions are given as pseudo-code in the prompt:
  `pickup(X)`, `chop()`, `put_on_plate()`, `place_on_counter()`, `pickup_plate()`,
  `deliver()`, `wait(n)`. A controller executes the skill with a shortest-path plan that is
  recomputed every step. The model is queried again only when the skill completes or one of
  its preconditions fails, in which case it is told which one. The reply has four lines:
  an analysis of the scene, the skill it thinks the partner is executing, its belief about
  the partner, and its plan (one skill).
- **`low`**: one call per environment step, the reply is a move (`left`, `right`, `up`,
  `down`, `interact`, `noop`). The models we tried often failed even the single-agent
  sub-tasks with this interface.

**Round memory** (`--memory`): a round is `--test-episodes` episodes with the same partner.
Within a round the model is shown what happened in the earlier episodes, which plays the
role of the interaction history of the learning agents; the memory is cleared when a new
round starts. `full` shows a summary of each earlier episode plus the step log of the
round, `summary` only the summaries, `none` nothing. Summaries are built from the game
events (outcome, pick-ups, chops, plating and delivery of both agents with time steps), not
by the model.

Level-1 partners act through their own network and receive their interaction history as
in their training.

## Running

Any OpenAI-compatible chat-completions endpoint works:

```bash
export LLM_BASE_URL=https://.../v1  LLM_API_KEY=...

# one run: the model in the left seat against one held-out Level-1 partner, 10 rounds x 5 episodes
python eval_llm.py --model <model id> --domain FC --actions high --memory full \
    --env-config environment/overcooked/config/fc_small_test.yaml --player-id 0 \
    --partner-ckpt logs/Overcooked/ppo_FC/bothsides/L1/Ours/seed16_seed16/ppo/30000000.pt \
    --test-pass 10 --test-episodes 5 --out-dir logs/Overcooked/llm_eval/demo

# against a Level-0 partner instead of --partner-ckpt: --l0-pair-type R1 --p 0.25
#   (a recipe R1..R6, or a type R1R2 / R3R4 / R5R6 with --l0-index 0|1 to choose the recipe of the type)
```

`scripts/overcooked/llm/run_llm_heldout.sh` runs the default evaluation: the model against
every held-out Level-1 partner, both seats in the homogeneous setting (FC) and the left seat
in the heterogeneous setting (FI, horizon 80). `MODEL`, `FC_POOL`, `FI_POOL`, `FC_SEEDS` and
`FI_PARTNERS` select the model and the partners. `run_llm_suite.sh` is a shorter survey
(Level-0 and Level-1 partners) for trying a new model.

Main arguments of `eval_llm.py`:

| argument | meaning |
|---|---|
| `--model`, `--temperature` (0), `--max-tokens`, `--timeout` | model and call settings |
| `--domain FC \| FI`, `--env-config`, `--player-id` | setting, kitchen config, seat of the model |
| `--partner-ckpt` or `--l0-pair-type`, `--l0-index`, `--p` | Level-1 checkpoint, or the Level-0 partner and its noise |
| `--test-pass`, `--test-episodes` | rounds and episodes per round |
| `--actions`, `--memory`, `--plan-len`, `--reason`, `--no-think` | interface and prompt options |
| `--out-dir`, `--trace-file` | output directory; optional file with every prompt, reply, action and resulting state |

## Output and checks

`<out-dir>/summary.json` holds the success rate, mean reward, per-episode results, the action
distributions of both agents and the call statistics. Check two things before using a run:

- **Failed or unparseable calls** (`llm.failures`, `llm.unparseable`). They fall back to
  doing nothing, so a run with many of them measures the endpoint, not the model.
- **Partner action entropy.** A value near `log 6 = 1.79` for a Level-1 partner means it is
  acting at random, i.e. the checkpoint was loaded with the wrong configuration.
