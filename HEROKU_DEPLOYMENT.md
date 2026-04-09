# HEROKU DEPLOYMENT GUIDE

## 🚀 Quick Start: Deploy in 5 Minutes

### Prerequisites

```bash
# 1. Heroku CLI installed
# Download from: https://devcenter.heroku.com/articles/heroku-cli

# 2. Git repository (✅ already done)
git status

# 3. Free PostgreSQL database (we'll use an external service)
```

---

## 📋 Step 1: Create Heroku App

```bash
# Login to Heroku
heroku login

# Create app (replace "worldcup-predictor" with your desired name)
heroku create worldcup-predictor

# Check it's created
heroku apps
```

Expected output:

```
Creating ⬢ worldcup-predictor... done
https://worldcup-predictor.herokuapp.com/ | https://git.heroku.com/worldcup-predictor.git
```

---

## 🗄️ Step 2: Setup PostgreSQL Database

### Option A: Free PostgreSQL (Render or Railway) - RECOMMENDED for Portfolio

```bash
# 1. Create free account at https://render.com or https://railway.app
# 2. Create PostgreSQL database
# 3. Get connection details: host, port, database, username, password
# 4. Note the connection string (you'll need it)
```

### Option B: Heroku Postgres Add-on (Legacy, now paid)

```bash
# If you have credits, you can use:
heroku addons:create heroku-postgresql:standard-0 --app worldcup-predictor
```

---

## 🔐 Step 3: Configure Environment Variables

Replace values from your PostgreSQL provider:

```bash
heroku config:set \
  POSTGRES_HOST=your-db.render.com \
  POSTGRES_PORT=5432 \
  POSTGRES_DB=worldcup_prod \
  POSTGRES_USER=your_user \
  POSTGRES_PASSWORD=your_secure_password \
  --app worldcup-predictor

# Verify they're set
heroku config --app worldcup-predictor
```

Expected output:

```
POSTGRES_HOST:             your-db.render.com
POSTGRES_PORT:             5432
POSTGRES_DB:               worldcup_prod
POSTGRES_PASSWORD:         ••••••••••••••••••
...
```

---

## ▶️ Step 4: Deploy to Heroku

```bash
# Add Heroku as git remote (if not already added)
heroku git:remote -a worldcup-predictor

# Deploy!
git push heroku main

# Watch the logs
heroku logs --tail --app worldcup-predictor
```

Expected output:

```
-----> Building on the Heroku platform
-----> Python app detected
-----> Installing requirements with pip
       ...
-----> Launching web dyno
       https://worldcup-predictor.herokuapp.com released
```

---

## 🧪 Step 5: Test the API

```bash
# Health check
curl https://worldcup-predictor.herokuapp.com/config

# Test prediction
curl -X POST https://worldcup-predictor.herokuapp.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Argentina",
    "away_team": "Brazil",
    "tournament": "World Cup Qualifiers"
  }'

# Open Swagger UI (interactive docs)
open https://worldcup-predictor.herokuapp.com/docs
```

Expected response:

```json
{
  "home_team": "Argentina",
  "away_team": "Brazil",
  "predicted_outcome": "home_win",
  "probabilities": {...},
  "match_segment": "worldcup",
  "is_override_triggered": false,
  ...
}
```

---

## 📊 Monitor Your App

```bash
# View logs
heroku logs --tail --app worldcup-predictor

# View metrics
heroku apps:info --app worldcup-predictor

# View config
heroku config --app worldcup-predictor

# Scale dynos (if needed)
heroku ps:scale web=2 --app worldcup-predictor
```

---

## 🐛 Troubleshooting

### Issue: "Procfile not found"

```bash
# Make sure Procfile is in root directory
ls -la Procfile

# Commit it
git add Procfile
git commit -m "add Heroku Procfile"
git push heroku main
```

### Issue: Database connection failed

```bash
# Check environment variables
heroku config --app worldcup-predictor

# Test connection locally with same vars
POSTGRES_HOST=your-host POSTGRES_USER=... python -c "from src.database.connection import get_sqlalchemy_engine; print(get_sqlalchemy_engine())"

# View app logs
heroku logs --tail --app worldcup-predictor
```

### Issue: "ModuleNotFoundError: No module named 'src'"

```bash
# Make sure requirements.txt includes all dependencies
pip freeze > requirements.txt
git add requirements.txt
git commit -m "update requirements"
git push heroku main
```

---

## 🎯 Your Live Portfolio Project

Once deployed, you have:

✅ **Live API**: https://worldcup-predictor.herokuapp.com/predict  
✅ **Swagger UI**: https://worldcup-predictor.herokuapp.com/docs  
✅ **GitHub Badge**: Add to README.md:

```markdown
[![Deployed on Heroku](https://www.herokucdn.com/deploy/button.svg)](https://worldcup-predictor.herokuapp.com/docs)
```

---

## 📈 Next Steps (Portfolio Enhancements)

1. **Add GitHub Star Button** to README
2. **Add API Status Badge** (https://shields.io/)
3. **Create Portfolio Post** about this project:
   - Architecture diagram
   - Tech stack highlights
   - Performance metrics
4. **Link from LinkedIn** to live API + GitHub repo
5. **Monitor on** https://www.heroku.com (free tier)

---

## 🧹 Cleanup (Optional)

If you want to delete the app later:

```bash
heroku apps:destroy worldcup-predictor --confirm worldcup-predictor
```

---

**Status**: Ready to deploy! 🚀
