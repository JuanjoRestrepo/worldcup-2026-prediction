---
name: web-devops
description: >
  Expert-level web development and DevOps skill. Use whenever the user mentions: scaffolding a
  web project (React, Next.js, Vue/Nuxt, Express, FastAPI, Django, MERN, PERN, T3 Stack); T3
  tools (tRPC, Prisma, Drizzle, NextAuth, Zod, Tailwind); Dockerfiles or Kubernetes configs;
  GitHub Actions or CI/CD pipelines; deploying to Vercel, Netlify, AWS, GCP, or Azure; or
  applying best practices (testing, security, observability, code quality). Trigger on vague
  requests too: "help me deploy this", "dockerize my app", "set up CI", "scaffold a project",
  "create a T3 app", "set up tRPC", "configure NextAuth", "review my pipeline". When in doubt,
  use this skill — it covers the full SDLC from project creation to production.
---

# Web Development & DevOps Skill

This skill makes Claude an expert web dev + DevOps engineer, capable of scaffolding projects,
writing infra configs, setting up CI/CD, deploying to cloud platforms, and applying full-SDLC
best practices — across a wide variety of stacks and platforms.

---

## How to Use This Skill

1. **Identify the task type** from the list below and read the matching section or reference file.
2. **Ask clarifying questions** if the user's stack or target platform is unclear — don't guess.
3. **Produce outputs** in the format most useful for the task: code files, configs, step-by-step
   runbooks, or a combination.
4. **Always apply** the best practices in the [Cross-Cutting Best Practices](#cross-cutting-best-practices)
   section regardless of the task type.
5. **Explain your decisions** — especially for infra and architecture choices. Don't just say *what*,
   say *why*. Users learn more and trust the output more when reasoning is visible.

### Quick Stack Decision Guide

| User wants... | Recommended stack |
|---|---|
| Full-stack app, fast MVP, type-safe end-to-end | **T3 Stack** (Next.js + tRPC + Prisma + NextAuth) |
| Full-stack with REST, widely familiar | **MERN / PERN** |
| Serverless-first, edge-optimized | **Next.js** on Vercel |
| API-only backend, Python team | **FastAPI** or **Django REST Framework** |
| High-traffic microservices | **Node.js/Express** + Docker + K8s |
| Containerized, cloud-agnostic | **Docker** + **GitHub Actions** + cloud of choice |

---

## Task Types

### 1. Project Scaffolding

When the user wants to start a new web project, create a proper structure. Ask:
- Framework/stack? (Next.js, T3, Vue/Nuxt, Express, FastAPI, Django, MERN, PERN)
- TypeScript or JavaScript? (default to TypeScript — strongly recommended)
- Will it be containerized? Deployed to which platform?
- Does it need auth, a database, an ORM?
- Monorepo or standalone? (suggest monorepo with Turborepo for T3/multi-package setups)

Then generate:
- Directory structure with explanation
- Core config files (`tsconfig.json`, `eslint.config.js`, `.env.example`, `.gitignore`, etc.)
- Package manager setup (`package.json` / `pyproject.toml` / `requirements.txt`)
- README with setup, environment variables, and deployment instructions

**T3 Stack — use `create-t3-app` as the baseline:**
```bash
pnpm create t3-app@latest
```
Covers: Next.js + TypeScript + tRPC + Prisma (or Drizzle) + NextAuth.js + Tailwind CSS + Zod.
See `references/scaffolding.md` for the full T3 project layout and key patterns.

See `references/scaffolding.md` for all stack-specific templates and patterns.

---

### 2. T3 Stack Deep Dive

The T3 Stack is an opinionated, type-safe full-stack framework. Understand its core pillars:

**tRPC** — end-to-end type-safe APIs without a schema:
- Define routers in `server/api/routers/`; compose in `server/api/root.ts`
- Use `publicProcedure` / `protectedProcedure` for auth-gated endpoints
- Prefer `useQuery` / `useMutation` from `@trpc/react-query` on the client
- Invalidate queries after mutations: `utils.post.getAll.invalidate()`

**Prisma** — the default ORM:
- Schema lives in `prisma/schema.prisma`; always run `prisma generate` after schema changes
- Use `prisma migrate dev` in development; `prisma migrate deploy` in CI/production
- Create a singleton client in `src/server/db.ts` to avoid connection pool exhaustion in dev

**Drizzle** — lighter alternative to Prisma, preferred for edge runtimes:
- Schema as TypeScript; fully type-safe queries without code generation at runtime
- Use with `drizzle-kit` for migrations; compatible with PlanetScale, Neon, Turso

**NextAuth.js (Auth.js v5)** — authentication:
- Config in `src/server/auth.ts`; wrap `app/layout.tsx` with `<SessionProvider>`
- Use `getServerAuthSession()` in server components; `useSession()` on the client
- Protect tRPC routes with `protectedProcedure` (checks session server-side)
- Providers: OAuth (GitHub, Google), credentials, magic links — configure in `authOptions`

**Zod** — runtime validation, used throughout:
- Validate tRPC inputs: `.input(z.object({...}))`
- Validate env vars: use `@t3-oss/env-nextjs` for type-safe environment variables
- Reuse schemas between client and server — single source of truth

**Tailwind CSS** — utility-first styling:
- Extend theme in `tailwind.config.ts`; use `cn()` utility (clsx + tailwind-merge) for conditional classes
- Pair with shadcn/ui for accessible, unstyled component primitives

**T3 Deployment to Vercel:**
- Set all env vars in Vercel dashboard (especially `DATABASE_URL`, `NEXTAUTH_SECRET`, `NEXTAUTH_URL`)
- Use a managed Postgres (PlanetScale, Neon, Supabase, Railway) — not a local DB
- Run `prisma migrate deploy` as a build step or via a one-off script post-deploy
- `NEXTAUTH_URL` must match the deployment URL exactly

**T3 + Docker (self-hosted):**
- T3 apps are Next.js apps — the standard Next.js multi-stage Dockerfile applies
- Set `output: "standalone"` in `next.config.js` for optimized Docker image size
- Run database migrations before starting the app container (init container or entrypoint script)

---

### 3. Docker & Kubernetes

When the user needs to containerize an app or deploy to a container orchestrator.

**Dockerfile best practices (always apply):**
- Use official, version-pinned base images (e.g., `node:20-alpine`, `python:3.12-slim`)
- Multi-stage builds: separate `builder` and `runner` stages to minimize image size
- Run as non-root user
- Use `.dockerignore` to exclude `node_modules`, `.env`, `.git`, build artifacts
- `COPY` only what's needed; layer order matters for cache efficiency
- Set `ENV NODE_ENV=production` or equivalent

**docker-compose:**
- Define services, volumes, and networks explicitly
- Use named volumes for persistence
- Use `depends_on` with `condition: service_healthy` where appropriate
- Never hardcode secrets — use environment variable substitution

**Kubernetes:**
- Always define `resources.requests` and `resources.limits`
- Use `Deployment` + `Service` + `Ingress` as the baseline
- Use `ConfigMap` for non-secret config, `Secret` for sensitive values
- Add `readinessProbe` and `livenessProbe`
- Use namespaces to separate environments

See `references/docker-kubernetes.md` for full templates.

---

### 4. GitHub Actions CI/CD

When the user wants to automate testing, building, or deploying via GitHub Actions.

**Pipeline structure (recommended):**
```
lint → test → build → deploy
```

**Best practices:**
- Pin action versions to a SHA or a major version tag (e.g., `actions/checkout@v4`)
- Store secrets in GitHub Secrets, never in YAML files
- Use `workflow_dispatch` for manual triggers alongside `push`/`pull_request`
- Cache dependencies: `actions/cache` for `node_modules`, pip, etc.
- Use matrix strategy for multi-version testing
- Add status badges to README
- Separate deploy jobs per environment (staging vs production) with `environment:` protection rules

**Common workflow templates:**
- Node.js app: lint → test → build → deploy to Vercel/cloud
- Python app: lint (ruff/flake8) → test (pytest) → build Docker → push to registry
- Docker: build → push to GHCR or ECR → deploy to K8s or ECS

See `references/github-actions.md` for full workflow YAML templates.

---

### 5. Cloud Deployment

#### Vercel / Netlify
- Framework auto-detection usually works; verify `build` and `output` settings
- Set environment variables in the platform dashboard — never commit `.env`
- Use preview deployments for PRs
- Configure custom domains and SSL via platform settings
- For Next.js / T3 on Vercel: prefer edge runtime for latency-sensitive routes; set `NEXTAUTH_URL` explicitly

#### AWS
- **Simple apps**: Amplify (like Vercel for AWS) or Elastic Beanstalk
- **Containers**: ECS Fargate (serverless containers) or EKS (Kubernetes)
- **Serverless**: Lambda + API Gateway; use SAM or CDK for infra-as-code
- Always use IAM roles with least privilege; never use root credentials
- Store secrets in AWS Secrets Manager or Parameter Store

#### GCP
- **Apps**: Cloud Run (best for containerized apps — serverless, scales to zero)
- **Kubernetes**: GKE Autopilot for managed K8s
- Use Workload Identity for GKE → GCP service auth (no service account key files)

#### Azure
- **Apps**: Azure App Service or Container Apps
- **Kubernetes**: AKS
- Use Managed Identity instead of connection strings where possible

#### Managed Databases (essential for T3 / full-stack apps)
| Provider | Best for | Notes |
|---|---|---|
| **Neon** | Serverless Postgres | Scales to zero, great for Vercel |
| **PlanetScale** | MySQL, high scale | Branching workflow, Drizzle-friendly |
| **Supabase** | Postgres + auth + storage | Self-hostable, includes Row Level Security |
| **Railway** | Simple Postgres/MySQL | Easiest setup for small projects |
| **Turso** | SQLite at edge | Drizzle-native, ultra-low latency reads |

---

## Cross-Cutting Best Practices

Apply these regardless of task type.

### Security
- Never commit secrets, API keys, or credentials — use `.env` + secret managers
- Validate env vars at startup with `@t3-oss/env-nextjs` (T3) or `pydantic-settings` (Python)
- Add `SECURITY.md` to new projects
- Set security headers (CSP, HSTS, X-Frame-Options) — use `helmet` for Node.js / `next-safe` for Next.js
- Dependency scanning: `npm audit`, `pip-audit`, Dependabot / Renovate
- Use HTTPS everywhere; enforce HTTPS redirects
- Validate and sanitize all user input server-side — use Zod for TS, Pydantic for Python
- Use parameterized queries / ORMs to prevent SQL injection
- Rotate secrets regularly; never log sensitive values

### Testing
- **Unit tests**: Vitest (T3/Vite-based), Jest (Node.js), pytest (Python)
- **Integration tests**: Supertest (Express), pytest + httpx (FastAPI), tRPC caller (T3 — test procedures directly without HTTP)
- **E2E tests**: Playwright (strongly preferred) or Cypress
- Aim for ≥80% coverage on business logic; don't chase 100% on boilerplate
- Run tests in CI on every PR; block merges on test failure
- For T3: mock the tRPC caller in unit tests; use `createCaller` from `appRouter`

### Observability
- **Logging**: Structured JSON logs; use `winston`/`pino` (Node) or `structlog` (Python)
- **Metrics**: Expose `/health` and `/metrics` endpoints; integrate with Prometheus/Grafana or cloud-native tooling
- **Tracing**: OpenTelemetry for distributed tracing (compatible with Vercel, Datadog, Honeycomb)
- **Error tracking**: Sentry (easy to add, works across all stacks including T3/Next.js)
- **Uptime monitoring**: at least one external ping (Better Uptime, UptimeRobot, Checkly)
- **Real User Monitoring**: Vercel Analytics or PostHog for product-level insights

### Code Quality
- Linting: ESLint with `@t3-oss/eslint-config` (T3), `eslint-config-airbnb` or `@antfu/eslint-config` (general); Ruff (Python)
- Formatting: Prettier (JS/TS), Ruff format (Python) — auto-format on save + in CI
- Pre-commit hooks: `husky` + `lint-staged` (JS/TS), `pre-commit` framework (Python)
- Type safety: TypeScript strict mode everywhere; Python type hints + `mypy` or `pyright`
- PR reviews: enforce via branch protection rules; require at least 1 approval
- Conventional Commits: use `commitlint` + `@commitlint/config-conventional` for consistent history and auto-changelogs

### Environment Management
- Always have at least: `development`, `staging`, `production`
- Use `.env.example` committed to repo; `.env` in `.gitignore`
- For T3: use `@t3-oss/env-nextjs` — validates env vars at build time, throws on missing values
- Load env vars at runtime, not build time, where possible
- Use a secret manager in production (AWS Secrets Manager, GCP Secret Manager, Doppler, Infisical)

### Performance
- Use React Server Components (RSC) for data-fetching in Next.js / T3 — avoid waterfalls
- Co-locate database queries with server components; avoid N+1 with `include`/`select` in Prisma
- Cache aggressively: Next.js `fetch` cache, React Query stale-while-revalidate, Redis for hot data
- Use `next/image` for optimized images; `next/font` for zero-layout-shift fonts
- Bundle analysis: `@next/bundle-analyzer` or `vite-bundle-visualizer`

---

## Output Format Guide

| Task | Primary Output |
|---|---|
| Project scaffold | Directory tree + key files as code blocks |
| T3 Stack setup | `create-t3-app` command + env vars + DB migration steps |
| Dockerfile | Single file with inline comments explaining each decision |
| K8s manifests | One YAML per resource, or kustomize layout |
| GitHub Actions | `.github/workflows/*.yml` file(s) |
| Cloud deploy | Step-by-step runbook + any config files |
| Code review | Inline suggestions + summary of issues by category |
| Best practices audit | Checklist with ✅/❌ + prioritized recommendations |
| Debugging help | Hypothesis → steps to verify → fix |

Always explain *why* a choice was made, not just *what* to do — especially for infra and architecture decisions.

---

## Reference Files

Read these when you need templates, full examples, or deeper patterns:

- `references/scaffolding.md` — Directory structures for all stacks including **T3**, Next.js, Express, FastAPI, MERN; essential config file templates
- `references/docker-kubernetes.md` — Dockerfile templates (Node.js, Python), docker-compose, Kubernetes manifests
- `references/github-actions.md` — Full CI/CD workflow YAMLs for Node.js, Python, Docker, ECS, multi-environment deploys