# CI/CD Pipeline Documentation

**Status:** ✅ Implemented & Ready  
**Trigger:** Push to `main`/`develop`, Pull Requests  
**Framework:** GitHub Actions

---

## Overview

The CI/CD pipeline automates blocking validation on every push and pull request. It ensures that:

- ✅ All unit tests pass
- ✅ dbt models are syntactically valid
- ✅ PostgreSQL-backed tests run against a real service container
- ✅ The repo installs reproducibly through `uv`

---

## Pipeline Stages

### 1. **Blocking Validation**

Runs on Ubuntu + PostgreSQL 15 service container.

**What it does:**

- Syncs dependencies with `uv`
- Sets up PostgreSQL with monitoring schema
- Runs the repository test suite without ignoring critical tests
- Runs `dbt parse` through `run_dbt.py`
- Comments on PRs if tests fail

**Tests Run:**

```bash
uv sync
uv run python -m pytest tests/ -v --tb=short
uv run python run_dbt.py parse
```

**Duration:** ~2-3 minutes

---

### 2. **Quality Hardening**

MyPy, Pylance-oriented cleanup, and stricter static analysis are the next step after functionality.
They are intentionally kept out of the blocking path until the functional validation layer is stable.

---

## Workflow Files

### `.github/workflows/ci.yml`

Main pipeline configuration:

- one blocking validation job
- PostgreSQL 15 service container
- `uv` environment sync
- pytest + dbt parse
- PR comment on failure

---

## How to Run Locally

Before pushing, run these locally to simulate CI:

```bash
# 1. Sync environment
uv sync

# 2. Start PostgreSQL
docker compose up -d postgres

# 3. Run blocking checks
uv run python -m pytest tests/ -v --tb=short
uv run python run_dbt.py parse
```

---

## GitHub Status Checks

When you push:

1. **Status checks appear** on your commit/PR
2. **Green ✅** = All tests passed, safe to merge
3. **Red ❌** = Tests failed, fix before merge
4. **Orange ⚠️** = Only for optional future hardening jobs

### Example Status:

```
✅ CI Pipeline / Test and dbt Validation — All checks passed
```

---

## PR Comments

If tests fail:

- Bot posts comment: "❌ CI Pipeline Failed  
  Please check the test logs above and fix any failures before merging."
- Comment links to failed job
- Author can re-run tests after fixes

---

## Blocked Merges (Enforce Standards)

To enforce CI passing before merge:

**On GitHub:**

1. Go to repo settings → Branches → Branch rules
2. Choose `main` branch
3. Enable: "Require status checks to pass before merging"
4. Select: "CI Pipeline / Test and dbt Validation"
5. Save

Now PRs can't merge until tests pass. ✅

---

## Environment Variables in CI

PostgreSQL connection in GitHub Actions:

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=worldcup_db
POSTGRES_USER=worldcup
POSTGRES_PASSWORD=worldcup_dev
```

These are set automatically by the `services.postgres` config.

---

## Extending the Pipeline

### Add a New Test Job

```yaml
new_job:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: My Custom Step
      run: echo "Custom test"
```

### Add Environment Variables

In `.github/workflows/ci.yml`:

```yaml
env:
  MY_VAR: my_value
```

### Add Manual Approval Gates

GitHub Actions → add `environment: production` for approval gates.

---

## Troubleshooting

| Problem                           | Solution                                                |
| --------------------------------- | ------------------------------------------------------- |
| Tests fail on CI but pass locally | Check env vars match `.env`                             |
| PostgreSQL connection fails       | Ensure postgres service is running (check logs)         |
| Python version mismatch           | CI uses `.python-version`, verify local `uv` interpreter |
| Actionsfail to trigger            | Check `.github/workflows/ci.yml` syntax with `yamllint` |
| Want to skip CI for a commit      | Use `[skip ci]` in commit message (⚠️ not recommended)  |

---

## Best Practices

✅ **DO:**

- Run tests locally before pushing
- Write tests first (TDD)
- Keep tests fast (<5 minutes)
- Use descriptive commit messages
- Review CI logs when tests fail

❌ **DON'T:**

- Disable CI checks
- Skip tests with "band-aid" fixes
- Push directly to `main` (use PRs + CI)
- Ignore failing blocking checks
- Leave failing tests in main branch

---

## Next Steps

Once pipeline is working:

1. **Enforce**: Require CI passing before PR merge (in branch settings)
2. **Extend**: Add MyPy and stricter Ruff once the functional layer is stable
3. **Monitor**: Watch for patterns in failures (flaky tests?)
4. **Optimize**: Keep `uv` cache warm in Actions
5. **Deploy**: Add deployment job for automated releases

---

## Cost & Performance

- **GitHub Actions**: Free for public repos (2000 min/month private)
- **Per-run cost**: ~2-3 minutes × ~40 runs/month = ~2-3 hours
- **Pipeline overhead**: ~$0 for public repos
- **Recommendation**: Enable when pushing to main (not on every branch)
