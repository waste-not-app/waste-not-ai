# Contributing

Thank you for helping improve WasteNot AI. This guide covers the workflow and standards every change is expected to follow.

## Workflow

1. Branch from `main` with a descriptive prefix: `feat/`, `fix/`, `refactor/`, `docs/`, or `chore/`.
2. Keep each commit to one logical change. Write a single-line message in the [Conventional Commits](https://www.conventionalcommits.org/) format, for example `fix(recommendation): reject evidence from expired guides`.
3. Run `make check` before pushing. CI runs `make ci`, which also audits dependencies.
4. Open a pull request against `main`. CI must pass before merging.

## Standards

- **Formatting and lint.** Format with `make fmt`. `make lint` must pass, including the Bandit security rules.
- **Dependencies.** Pin every dependency with hashes in the lock files. Never install unpinned packages in CI or on the host.
- **Contracts.** A contract change requires `make export-contracts`, a review of the generated diff, and a paired update of the pin in the backend. See [docs/INTEGRATION.md](docs/INTEGRATION.md).
- **Guides.** Never publish a draft guide or invent reviewer metadata. Only `PUBLISHED` guides with a real review reach production.
- **Data.** Never commit photos, datasets, archives, trained weights, or API keys.
- **Comments.** Comment only on what the code cannot express.

## Security

Never commit `.env` files, Gemini keys, or service tokens. Report vulnerabilities privately to the maintainers rather than in a public issue.
