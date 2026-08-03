# Anonymous Sync Setup

This repo auto-syncs to [anonymous.4open.science](https://anonymous.4open.science) on every push to `main`, via the GitHub Action in `.github/workflows/sync-anonymous.yml`.

## How it works

anonymous.4open.science holds a **read-only mirror** of this GitHub repo. It does NOT accept git pushes. To update the mirror, you trigger a refresh:

```
git push origin main                              # 1. push to GitHub
       ↓
GitHub Action triggers                            # 2. automatic on push
       ↓
POST /api/repo/<id>/refresh with Bearer token     # 3. calls anonymous API
       ↓
anonymous.4open.science re-pulls from GitHub      # 4. mirror updated
       ↓
Public URL content refreshes within minutes       # 5. reviewer sees latest
```

## One-time setup (3 steps, ~2 minutes)

### 1. Generate an anonymous.4open.science API token

- Go to <https://anonymous.4open.science> and sign in with GitHub.
- Open your **profile / account settings → API tokens** (or visit <https://anonymous.4open.science/user>).
- Click **Generate new token**, give it a name like `rcp-audit-artifact-ci`.
- **Copy the token immediately** — it's shown only once.

### 2. Add the token as a GitHub Secret

- In this repo: **Settings → Secrets and variables → Actions → New repository secret**
- **Name:** `ANONYMOUS_API_TOKEN`
- **Value:** paste the token from step 1

### 3. (Optional) Set the repo ID variable

The default anonymous repo ID is `rcp-audit-artifact-B314` (hardcoded in the workflow as fallback). If yours differs:

- **Settings → Secrets and variables → Actions → Variables → New repository variable**
- **Name:** `ANONYMOUS_REPO_ID`
- **Value:** your repo ID (the part after `/r/` in the anonymous URL)

## Verify

1. Make any commit and `git push origin main`.
2. Open the **Actions** tab; you should see "Sync to anonymous.4open.science" running.
3. After green check, open <https://anonymous.4open.science/r/rcp-audit-artifact-B314/README.md> and hard-refresh. New content should appear within minutes.

## Manual trigger (no push needed)

**Actions → Sync to anonymous.4open.science → Run workflow**

Useful if the GitHub push happened before you set up the token, or to recover from a transient failure.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ANONYMOUS_API_TOKEN secret is not set` | Secret not added | Do step 2 above |
| HTTP 401 Unauthorized | Token invalid/expired | Regenerate at anonymous.4open.science profile |
| HTTP 403 Forbidden | Token owner isn't repo owner/coauthor | Use a token from the account that owns `rcp-audit-artifact-B314` |
| HTTP 404 Not Found | Wrong `ANONYMOUS_REPO_ID` | Update the variable; default is `rcp-audit-artifact-B314` |
| HTTP 409 Conflict | Repo in preparing/removing/expiring state | Wait a minute, retry |
| Action succeeds but anonymous URL shows old content | Browser/CDN cache | Hard-refresh (Ctrl+Shift+R); wait 5 min; try incognito |
| Refresh succeeds but new files don't appear | Push to GitHub not yet fetched by anonymous | Confirm `git push origin main` succeeded; wait for anonymous re-pull |

## Removing the sync

- Delete `.github/workflows/sync-anonymous.yml`
- Delete the `ANONYMOUS_API_TOKEN` secret
- (Optional) Revoke the API token at anonymous.4open.science profile

## Source / credits

- anonymous.4open.science API endpoint: [`POST /api/repo/:repoId/refresh`](https://github.com/tdurieux/anonymous_github/blob/master/src/server/routes/repository-private.ts) — calls `repo.updateIfNeeded({ force: true })`.
- Bearer token auth: [`bearerTokenAuth`](https://github.com/tdurieux/anonymous_github/blob/master/src/server/routes/token-auth.ts) middleware, matches `Authorization: Bearer <token>` against hashed `apiTokens` in user model.
