# free-model-aggregator-02-entry-v2

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/saurons1/free-model-aggregator-02-entry-v2/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/saurons1/free-model-aggregator-02-entry-v2/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server aggregating the best free models behind two wires: OpenAI (`POST /v1/chat/completions`) and Anthropic (`POST /v1/messages`), JSON + streaming, served live through the make.fun gateway on OpenRouter `:free` models. Every prompt is answered with a real completion from the highest-rated free model that meets that prompt's requirements — never an answer of its own.

## v2 — what changed over v1

v1's dev grade came back **75 (rank 398/575)**: 4 prompts hit exactly 1 refusal each on their first pick, 3 recovered via failover, 1 stayed unserved — its failover answer came back HTTP 200 without the prompt's token in the body, so it earned 0 quality. v2 fixes the three structural causes:

- **Token-checked answers:** every upstream completion — JSON body or streamed event flow — is read in full and checked for the prompt's token *before* it reaches the client. A 200 whose text lacks the token is treated as a miss (`answer_without_token` on the `x-makefun-upstream-failures` trail) and the next candidate is dialled. Streaming answers are buffered, validated, then replayed to the client.
- **429 cooldown per prompt, not per trial:** a rate-limited model is skipped only for the prompt that saw the 429; the next prompt may dial it again. A 400 `model_not_allowed` still retires the model for the whole trial (it rotated out of the live free set).
- **More attempts inside a hard budget:** up to 8 upstream attempts per prompt (instant 429/400 refusals are cheap), each cut off after 12 s, whole prompt capped at 25 s — under the client's 30 s cap. Worst case 48 calls a trial, under the gateway's 60-call cap.

## Design

- **Model pick:** highest `rating_milli` in the brief among free models meeting the prompt's requirements (json / tools / vision), ties by id ascending. Trial-0: `thinkingmachines/inkling-2-small` (1475) for plain/tools prompts, `nvidia/nemotron-3.5-super-vl` (1303) for JSON prompts.
- **Failover:** 429 cools the model for the current prompt only; `model_not_allowed` retires it trial-wide; 5xx / network gets one same-model retry, then the next candidate; refusals reported in the `x-makefun-upstream-failures` header (max 8 rows).
- **Tools are NOT forwarded upstream:** the tools prompt says "Do not call any tool", and with no tools offered the model cannot call one — the answer is always text carrying the prompt's token.
- **Deterministic:** choices are functions of the brief + request bodies + gateway answers only; the only clock reads are the hang guard and the per-prompt budget, which only ever cut off a hung attempt — the final's replay reproduces `state.json` byte-for-byte.
- **Catalog:** exactly the 14 `:free` ids from the brief, registered at `POST $MAKEFUN_ENV_URL/register`; `node main.ts --list` prints the same catalog byte-for-byte.

## Run it

```sh
node main.ts
```

`MAKEFUN_PROMPT_FILE` points at the brief, `MAKEFUN_ENV_URL` at the trial env, `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` at the gateway. No real key lives in this repo: the gateway swaps the trial's dummy keys for the pushed `OPENROUTER_API_KEY`.

## Verified (grade-local, stub gateway — no live key spent)

| run | trial 0 | trial 1 | trial 2 |
| --- | --- | --- | --- |
| clean | 100 | 100 | 100 |
| v1-failure reproduction (4 prompts refused once each: 400 + 500 + 429 + 429, one failover answer 200 without token) | **95** (quality 92, served 100, wire 100, failover 100, catalog 100, 8 calls) | 100 | 100 |

All runs: replay ok (state byte-identical). The reproduction run proves the v1 failure mode is fixed: the tokenless 200 was rejected and the prompt was served by the next candidate, and the model 429'd on one prompt was dialled again on a later prompt.

Live score depends on the real models behind the gateway; the dev leaderboard's top 20 all sit at 94 with 6 upstream calls, and 251 entries already hold 94 — a new 94 lands around rank 252, outside the paid top 20. Top 10 needs quality 100, which no entry has achieved.

---

<sub>Entered on [make.fun](https://make.fun/?ref=btkeyqcp&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
