# Pilot Guides

`pilot-source-reviewed-v6.json` contains the four guides used by the production recommendation worker:

- PET bottles;
- HDPE containers;
- non-aerosol aluminium cans;
- cardboard.

They implement the [v6 proposals](../guide-proposals/README.md).

## Review status

- **Review method.** The sources were checked by an automated review, which the repository owner delegated. The metadata states this explicitly: **"Codex automated source review; not human safety approval"**. It is not a signature or a human safety review.
- **Validity.** Checked on 4 November 2026 and valid until 4 December 2026, a 30-day pilot review cycle. The worker rejects expired guides automatically.
- **Other guides.** The remaining 74 guides stay unpublished as drafts. The original draft and proposal files were never converted into published guides; this artifact carries its own publication metadata.

## Scope

The guides apply to users in Indonesia who confirm three things:

- the material and condition;
- the item's former contents, where relevant;
- that a local collector accepts that material in that condition.

Foreign sources support the material principles, not Indonesian law or facility acceptance. These items must never use the pilot rules:

- aerosol or pressurized cans;
- containers with a chemical history;
- materials outside the pilot scope.

The guides make no claim about the material from a photo, or about refill safety.

## Installation

1. Install the file with its checksum at a versioned path that only the recommendation service can read.
2. Set `AI_GUIDES_PATH` to that path. The worker validates the schema, status, and validity window on load.
3. Keep `AI_ALLOW_PROTOTYPE=false` and the real Gemini provider.
4. Enable the recommendation capability only after a worker check and one end-to-end request through the public API both pass.

Slow or invalid provider responses still return `UNAVAILABLE`. The server never invents a method or saves anything automatically.

The automated source review does not replace any of the following:

- a model holdout evaluation;
- human review of risky handling;
- LLM quality and latency checks;
- Android acceptance testing;
- approval for a general launch.
