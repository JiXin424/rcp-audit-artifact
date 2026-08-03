# Anonymous Sync Setup

This repo auto-syncs to [anonymous.4open.science](https://anonymous.4open.science) on every push to `main`, via the GitHub Action in `.github/workflows/sync-anonymous.yml`.

## One-time setup

1. **Get the anonymous push URL.**
   - Go to https://anonymous.4open.science and sign in.
   - Open (or create) the anonymous repo that maps to this GitHub repo.
   - Look for the "git push" or "update" URL; it has the form:
     ```
     https://anonymous:TOKEN@anonymous.4open.science/REPO_ID.git
     ```
     where `TOKEN` is the long credential and `REPO_ID` is the 4-char ID (e.g., `B314`).

2. **Add the URL as a GitHub Secret.**
   - In this repo: **Settings → Secrets and variables → Actions → New repository secret**
   - **Name:** `ANONYMOUS_PUSH_URL`
   - **Value:** the full URL from step 1 (including `https://anonymous:TOKEN@...`)

3. **Push to `main`.** The Action triggers automatically.

## Verifying

- After each push, check the **Actions** tab for green check.
- The public anonymous URL stays the same (e.g., `https://anonymous.4open.science/r/rcp-audit-artifact-B314/README.md`) and shows the latest content.

## Manual trigger

If you need to re-sync without a push:
- **Actions → Sync to anonymous.4open.science → Run workflow**

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ANONYMOUS_PUSH_URL secret is not set` | Do step 2 above |
| `Authentication failed` | Re-fetch the URL from anonymous.4open.science; tokens may rotate |
| `Repository not found` | The REPO_ID in the URL may be wrong; verify on anonymous.4open.science |
| Action runs but anonymous URL shows old content | anonymous.4open.science caches; hard-refresh or wait a few minutes |

## Removing the sync

Delete `.github/workflows/sync-anonymous.yml` and the `ANONYMOUS_PUSH_URL` secret.
