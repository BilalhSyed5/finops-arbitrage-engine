# Configuration

The engine ships with defaults in `finops/data/`:

| File | What it holds |
|---|---|
| `model_pricing.json` | Token prices for ~55 OpenAI, Anthropic, Google and open-weight models: input, cached input, cache write, output, Batch discount, long-prompt surcharges, successor models, announced price changes. Each snapshot carries an `as_of` date; reports warn after 45 days. |
| `ec2_pricing.json` | EC2 on-demand rates (us-east-1) with vCPU and GPU counts, used for right-sizing. |
| `routing_policy.json` | Which task types may move to cheaper models and to which ones, plus caching / batch / proof thresholds. |

You rarely edit those. Instead, pass overrides that are **merged over** them:

* `--pricing-config DIR`: a folder with `model_pricing.json` and/or `ec2_pricing.json`
  holding only the entries that differ (negotiated rates, savings plans, fine-tuned
  models). See [`example_pricing_override/`](example_pricing_override/).
* `--policy FILE`: change routing targets, allow cross-provider moves, add task types,
  remove one (`"summarization": null`), or raise the proof bar. See
  [`example_policy.json`](example_policy.json).
* `--attribution FILE`: map project ids / API keys / models to team, feature, customer
  and task type. See [`../sample_data/attribution.json`](../sample_data/attribution.json).

Check what a model name resolves to with `finops pricing <model> ...`, and list the
whole book with `finops pricing --list` (or `--list anthropic`).
