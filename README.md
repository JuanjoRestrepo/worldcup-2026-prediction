# World Cup 2026 Data Pipeline

## Overview

Data engineering pipeline for ingesting, processing, and modeling international football data for predictive analysis.

## Stack

- Python
- PostgreSQL
- Docker

## Setup

```bash
docker compose up -d
```

If Docker Desktop still returns an `EOF` while creating the container after these compose fixes,
the remaining issue is usually local Docker engine state rather than the project files. Restart
Docker Desktop and retry the command.
