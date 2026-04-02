# CI/CD Pipeline Documentation

**Status:** ✅ Implemented & Ready  
**Trigger:** Push to `main`/`develop`, Pull Requests  
**Framework:** GitHub Actions

---

## Overview

The CI/CD pipeline automates testing and code quality checks on every push and pull request. It ensures that:

- ✅ All unit tests pass
- ✅ Code quality standards are met
- ✅ dbt models are syntactically valid
- ✅ Security vulnerabilities are detected
- ✅ Type hints are valid (optional)

---

## Pipeline Stages

### 1. **Test Suite** (Primary)

Runs on Ubuntu + PostgreSQL 15 service container.

**What it does:**

- Installs dependencies from `requirements.txt`
- Sets up PostgreSQL with monitoring schema
- Runs pytest on core modules:
  - `test_team_aliases.py` (13 tests) ✅
  - `test_api_hardening.py` ✅
  - Other integration tests (skipped in CI for now)
- Comments on PRs if tests fail

**Tests Run:**

```bash
pytest tests/ -v --tb=short \
  --ignore=tests/test_inference_logger.py   # Needs live DB
  --ignore=tests/test_api.py                # Requires full setup
  --ignore=tests/test_database_persistence.py
```

**Duration:** ~2-3 minutes

---

### 2. **Code Quality Checks** (Secondary)

Runs linting and security checks.

**What it does:**

- Ruff: Python style checker (non-blocking)
- MyPy: Type checking (non-blocking)
- Bandit: Security scanning (non-blocking)
- dbt parse: Model syntax validation (non-blocking)

**All code quality checks are advisory** (run with `|| true`) so they don't block merges, but failures are visible in logs.

---

## Workflow Files

### `.github/workflows/ci.yml`

Main pipeline configuration (2 jobs):

**Job 1: `test`**

- Runs unit tests
- Services: PostgreSQL 15
- Publishes results + PR comments
- **Status:** Blocks if tes fail ❌

**Job 2: `lint`**

- Code quality checks
- No blocking (all checks use `|| true`)
- **Status:** Advisory only ⚠️

---

## How to Run Locally

Before pushing, run these locally to simulate CI:

```bash
# 1. Activate venv
.venv\Scripts\activate

# 2. Run tests (skip DB-dependent tests)
pytest tests/test_team_aliases.py tests/test_api_hardening.py -v

# 3. Check code quality (optional)
pip install ruff mypy bandit
ruff check src/
mypy src/ --ignore-missing-imports
bandit -ll -r src/
```

---

## GitHub Status Checks

When you push:

1. **Status checks appear** on your commit/PR
2. **Green ✅** = All tests passed, safe to merge
3. **Red ❌** = Tests failed, fix before merge
4. **Orange ⚠️** = Quality warnings (for info only)

### Example Status:

```
✅ CI Pipeline / Test Suite — All checks passed
⚠️  CI Pipeline / Code Quality Checks — Some non-critical warnings
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
4. Select: "CI Pipeline / Test Suite"
5. Save

Now PRs can't merge until tests pass. ✅

---

## Environment Variables in CI

PostgreSQL connection in GitHub Actions:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=worldcup_db
DB_USER=worldcup
DB_PASSWORD=worldcup_dev
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
| Python version mismatch           | CI uses Python 3.13, verify local version               |
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
- Ignore security warnings
- Leave failing tests in main branch

---

## Next Steps

Once pipeline is working:

1. **Enforce**: Require CI passing before PR merge (in branch settings)
2. **Extend**: Add more tests as features are added
3. **Monitor**: Watch for patterns in failures (flaky tests?)
4. **Optimize**: Cache dependencies to speed up runs
5. **Deploy**: Add deployment job for automated releases

---

## Cost & Performance

- **GitHub Actions**: Free for public repos (2000 min/month private)
- **Per-run cost**: ~2-3 minutes × ~40 runs/month = ~2-3 hours
- **Pipeline overhead**: ~$0 for public repos
- **Recommendation**: Enable when pushing to main (not on every branch)
