from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


def write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13.5",
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "codemirror_mode": {"name": "ipython", "version": 3},
        },
    }
    path.write_text(nbf.writes(nb), encoding="utf-8")


nb01_cells = []
nb02_cells = []
nb03_cells = []


nb01_cells = [
    md(
        """
        # 01. Data Foundation

        Construye la capa base del proyecto:
        - ingesta del historico internacional
        - estandarizacion de columnas
        - validacion de calidad
        - creacion de targets basicos
        - exportacion del dataset limpio para modelado
        """
    ),
    code(
        """
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        pd.set_option("display.max_columns", None)
        sns.set_style("whitegrid")

        rawPath = Path("../data/raw/international_results.csv")
        cleanPath = Path("../data/clean/clean_matches.csv")
        rawUrl = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

        print("Raw path:", rawPath)
        print("Clean path:", cleanPath)
        """
    ),
    md("## 1. Carga del historico"),
    code(
        """
        if rawPath.exists():
            rawDf = pd.read_csv(rawPath)
            sourceUsed = str(rawPath)
        else:
            rawDf = pd.read_csv(rawUrl)
            rawPath.parent.mkdir(parents=True, exist_ok=True)
            rawDf.to_csv(rawPath, index=False)
            sourceUsed = rawUrl

        rawDf = rawDf.rename(
            columns={
                "home_team": "homeTeam",
                "away_team": "awayTeam",
                "home_score": "homeScore",
                "away_score": "awayScore",
            }
        )
        rawDf["date"] = pd.to_datetime(rawDf["date"])
        rawDf = rawDf.sort_values("date", kind="mergesort").reset_index(drop=True)

        print("Source used:", sourceUsed)
        print("Shape rawDf:", rawDf.shape)
        rawDf.head()
        """
    ),
    md("## 2. Validacion de calidad"),
    code(
        """
        requiredColumns = [
            "date",
            "homeTeam",
            "awayTeam",
            "homeScore",
            "awayScore",
            "tournament",
            "city",
            "country",
            "neutral",
        ]

        missingColumns = [col for col in requiredColumns if col not in rawDf.columns]
        assert not missingColumns, f"Faltan columnas requeridas: {missingColumns}"
        assert rawDf[requiredColumns].isnull().sum().sum() == 0, "Hay nulos en columnas criticas"
        assert (rawDf[["homeScore", "awayScore"]] < 0).sum().sum() == 0, "Hay marcadores negativos"

        duplicateRows = rawDf.duplicated().sum()
        duplicateMatchKeys = rawDf.duplicated(subset=["date", "homeTeam", "awayTeam"]).sum()

        print("Duplicados exactos:", duplicateRows)
        print("Duplicados por llave date-home-away:", duplicateMatchKeys)
        print("Rango de fechas:", rawDf["date"].min(), "->", rawDf["date"].max())
        print("Numero de torneos unicos:", rawDf["tournament"].nunique())
        """
    ),
    md("## 3. Targets basicos"),
    code(
        """
        rawDf["matchResult"] = np.select(
            [
                rawDf["homeScore"] > rawDf["awayScore"],
                rawDf["homeScore"] < rawDf["awayScore"],
            ],
            [
                "win",
                "loss",
            ],
            default="draw",
        )
        rawDf["totalGoals"] = rawDf["homeScore"] + rawDf["awayScore"]

        print(rawDf["matchResult"].value_counts(normalize=True).round(4))
        rawDf.head()
        """
    ),
    md("## 4. Resumen exploratorio"),
    code(
        """
        summaryDf = pd.DataFrame(
            {
                "metric": [
                    "matches",
                    "teams",
                    "tournaments",
                    "avgHomeGoals",
                    "avgAwayGoals",
                    "avgTotalGoals",
                ],
                "value": [
                    len(rawDf),
                    pd.Index(rawDf["homeTeam"]).union(rawDf["awayTeam"]).nunique(),
                    rawDf["tournament"].nunique(),
                    rawDf["homeScore"].mean(),
                    rawDf["awayScore"].mean(),
                    rawDf["totalGoals"].mean(),
                ],
            }
        )

        summaryDf
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        rawDf["matchResult"].value_counts().plot(kind="bar", ax=axes[0], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        axes[0].set_title("Distribucion de resultados")
        axes[0].set_xlabel("")

        rawDf["totalGoals"].hist(bins=30, ax=axes[1], color="#4c78a8")
        axes[1].set_title("Distribucion de goles totales")

        rawDf.groupby(rawDf["date"].dt.year).size().plot(ax=axes[2], color="#f58518")
        axes[2].set_title("Partidos por anio")
        axes[2].set_xlabel("Year")

        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 5. Filtro temporal y exportacion"),
    code(
        """
        analysisStartDate = pd.Timestamp("2000-01-01")

        cleanDf = rawDf.loc[rawDf["date"] >= analysisStartDate].copy()
        cleanDf = cleanDf.sort_values("date", kind="mergesort").reset_index(drop=True)

        assert cleanDf.duplicated(subset=["date", "homeTeam", "awayTeam"]).sum() == 0, "Hay duplicados en el dataset limpio"
        assert cleanDf[requiredColumns + ["matchResult", "totalGoals"]].isnull().sum().sum() == 0, "Hay nulos luego de la limpieza"

        cleanPath.parent.mkdir(parents=True, exist_ok=True)
        cleanDf.to_csv(cleanPath, index=False)

        print("Shape cleanDf:", cleanDf.shape)
        print("Rango limpio:", cleanDf["date"].min(), "->", cleanDf["date"].max())
        print("Archivo exportado en:", cleanPath)
        cleanDf.head()
        """
    ),
]


nb02_cells = [
    md(
        """
        # 02. Feature Engineering

        Objetivo:
        - generar features pre-partido sin fuga de informacion
        - construir ratings dinamicos y forma reciente
        - exportar un dataset maestro para modelado probabilistico
        - exportar una version final mas limpia para referencia
        """
    ),
    code(
        """
        from collections import defaultdict, deque
        from pathlib import Path

        import numpy as np
        import pandas as pd

        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", 200)

        cleanPath = Path("../data/clean/clean_matches.csv")
        modelingBasePath = Path("../data/intermediate/modeling_base_dataset.csv")
        finalModelPath = Path("../data/finalModelDf/finalModelDFmodeling_dataset.csv")

        print("Clean path:", cleanPath)
        print("Modeling base path:", modelingBasePath)
        print("Final model path:", finalModelPath)
        """
    ),
    md("## 1. Carga del dataset limpio"),
    code(
        """
        df = pd.read_csv(cleanPath)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

        print("Shape clean matches:", df.shape)
        df.head()
        """
    ),
    md("## 2. Validacion previa"),
    code(
        """
        requiredColumns = [
            "date",
            "homeTeam",
            "awayTeam",
            "homeScore",
            "awayScore",
            "tournament",
            "city",
            "country",
            "neutral",
            "matchResult",
            "totalGoals",
        ]

        missingColumns = [col for col in requiredColumns if col not in df.columns]
        assert not missingColumns, f"Faltan columnas requeridas: {missingColumns}"
        assert df[requiredColumns].isnull().sum().sum() == 0, "Hay nulos en columnas requeridas"
        assert (df[["homeScore", "awayScore"]] < 0).sum().sum() == 0, "Hay marcadores negativos"
        assert df.duplicated(subset=["date", "homeTeam", "awayTeam"]).sum() == 0, "Hay llaves duplicadas de partido"

        print("Rango de fechas:", df["date"].min(), "->", df["date"].max())
        print("Distribucion del target:")
        print(df["matchResult"].value_counts(normalize=True).round(4))
        """
    ),
    md("## 3. Funciones auxiliares"),
    code(
        """
        def safeMean(values):
            if len(values) == 0:
                return np.nan
            return float(np.mean(values))


        def safeRate(numerator, denominator):
            if denominator == 0:
                return np.nan
            return float(numerator / denominator)


        def getResultPoints(homeScore, awayScore):
            if homeScore > awayScore:
                return 3, 0
            if homeScore < awayScore:
                return 0, 3
            return 1, 1


        def categorizeTournament(tournament):
            tournamentText = str(tournament).strip().lower()

            if any(keyword in tournamentText for keyword in ["world cup", "fifa world cup", "mundial"]):
                return "worldCup"
            if any(keyword in tournamentText for keyword in ["qualification", "qualifier", "qualifiers", "eliminatoria", "eliminatorias"]):
                return "qualification"
            if any(keyword in tournamentText for keyword in ["friendly", "amistoso", "amistosos"]):
                return "friendly"

            continentalKeywords = [
                "copa america",
                "conmebol",
                "euro",
                "european championship",
                "african cup",
                "afcon",
                "asian cup",
                "gold cup",
                "concacaf",
                "nations league",
                "uefa nations",
                "oceania",
                "ofc",
            ]
            if any(keyword in tournamentText for keyword in continentalKeywords):
                return "continental"

            return "other"


        def createTeamState():
            return {
                "elo": 1500.0,
                "matchesPlayed": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goalsFor": 0,
                "goalsAgainst": 0,
                "lastMatchDate": None,
                "last5Points": deque(maxlen=5),
                "last5GoalsFor": deque(maxlen=5),
                "last5GoalsAgainst": deque(maxlen=5),
                "last10Points": deque(maxlen=10),
                "last10GoalsFor": deque(maxlen=10),
                "last10GoalsAgainst": deque(maxlen=10),
            }


        def createPairState():
            return {
                "matchesPlayed": 0,
                "firstTeamWins": 0,
                "secondTeamWins": 0,
                "draws": 0,
            }


        def getPairKey(teamA, teamB):
            return tuple(sorted([teamA, teamB]))


        baseK = 20
        homeAdvantage = 65


        def expectedScore(ratingA, ratingB):
            return 1 / (1 + 10 ** ((ratingB - ratingA) / 400))


        def updateElo(homeElo, awayElo, homeScore, awayScore, neutral=False):
            adjustedHomeElo = homeElo if neutral else homeElo + homeAdvantage
            expectedHome = expectedScore(adjustedHomeElo, awayElo)

            if homeScore > awayScore:
                actualHome = 1.0
            elif homeScore < awayScore:
                actualHome = 0.0
            else:
                actualHome = 0.5

            newHomeElo = homeElo + baseK * (actualHome - expectedHome)
            newAwayElo = awayElo + baseK * ((1 - actualHome) - (1 - expectedHome))
            return newHomeElo, newAwayElo, expectedHome
        """
    ),
    md("## 4. Generacion del dataset de modelado"),
    code(
        """
        teamState = defaultdict(createTeamState)
        pairState = defaultdict(createPairState)
        featureRows = []

        for row in df.itertuples(index=False):
            matchDate = row.date
            homeTeam = row.homeTeam
            awayTeam = row.awayTeam
            homeScore = int(row.homeScore)
            awayScore = int(row.awayScore)
            tournament = row.tournament
            neutralFlag = int(bool(row.neutral))

            homeState = teamState[homeTeam]
            awayState = teamState[awayTeam]

            homeElo = homeState["elo"]
            awayElo = awayState["elo"]
            eloDiff = homeElo - awayElo
            absEloDiff = abs(eloDiff)
            eloExpectedHomeWin = expectedScore(homeElo if neutralFlag else homeElo + homeAdvantage, awayElo)

            homeMatchesPlayed = homeState["matchesPlayed"]
            awayMatchesPlayed = awayState["matchesPlayed"]

            homeWinRate = safeRate(homeState["wins"], homeMatchesPlayed)
            awayWinRate = safeRate(awayState["wins"], awayMatchesPlayed)
            homeDrawRate = safeRate(homeState["draws"], homeMatchesPlayed)
            awayDrawRate = safeRate(awayState["draws"], awayMatchesPlayed)
            homeLossRate = safeRate(homeState["losses"], homeMatchesPlayed)
            awayLossRate = safeRate(awayState["losses"], awayMatchesPlayed)

            homeGoalsForAvg = safeRate(homeState["goalsFor"], homeMatchesPlayed)
            awayGoalsForAvg = safeRate(awayState["goalsFor"], awayMatchesPlayed)
            homeGoalsAgainstAvg = safeRate(homeState["goalsAgainst"], homeMatchesPlayed)
            awayGoalsAgainstAvg = safeRate(awayState["goalsAgainst"], awayMatchesPlayed)
            homeGoalDiffAvg = safeRate(homeState["goalsFor"] - homeState["goalsAgainst"], homeMatchesPlayed)
            awayGoalDiffAvg = safeRate(awayState["goalsFor"] - awayState["goalsAgainst"], awayMatchesPlayed)

            homeLast5WinRate = safeMean([1 if points == 3 else 0 for points in homeState["last5Points"]])
            awayLast5WinRate = safeMean([1 if points == 3 else 0 for points in awayState["last5Points"]])
            homeLast5PointsAvg = safeMean(homeState["last5Points"])
            awayLast5PointsAvg = safeMean(awayState["last5Points"])
            homeLast5GoalsForAvg = safeMean(homeState["last5GoalsFor"])
            awayLast5GoalsForAvg = safeMean(awayState["last5GoalsFor"])
            homeLast5GoalsAgainstAvg = safeMean(homeState["last5GoalsAgainst"])
            awayLast5GoalsAgainstAvg = safeMean(awayState["last5GoalsAgainst"])
            homeLast5GoalDiffAvg = safeMean([gf - ga for gf, ga in zip(homeState["last5GoalsFor"], homeState["last5GoalsAgainst"])])
            awayLast5GoalDiffAvg = safeMean([gf - ga for gf, ga in zip(awayState["last5GoalsFor"], awayState["last5GoalsAgainst"])])

            homeLast10PointsAvg = safeMean(homeState["last10Points"])
            awayLast10PointsAvg = safeMean(awayState["last10Points"])
            homeLast10GoalsForAvg = safeMean(homeState["last10GoalsFor"])
            awayLast10GoalsForAvg = safeMean(awayState["last10GoalsFor"])
            homeLast10GoalsAgainstAvg = safeMean(homeState["last10GoalsAgainst"])
            awayLast10GoalsAgainstAvg = safeMean(awayState["last10GoalsAgainst"])

            homeDaysSinceLastMatch = (matchDate - homeState["lastMatchDate"]).days if homeState["lastMatchDate"] is not None else np.nan
            awayDaysSinceLastMatch = (matchDate - awayState["lastMatchDate"]).days if awayState["lastMatchDate"] is not None else np.nan

            pairKey = getPairKey(homeTeam, awayTeam)
            pairStats = pairState[pairKey]
            if pairStats["matchesPlayed"] > 0:
                if homeTeam == pairKey[0]:
                    homeH2HWinRate = pairStats["firstTeamWins"] / pairStats["matchesPlayed"]
                    awayH2HWinRate = pairStats["secondTeamWins"] / pairStats["matchesPlayed"]
                else:
                    homeH2HWinRate = pairStats["secondTeamWins"] / pairStats["matchesPlayed"]
                    awayH2HWinRate = pairStats["firstTeamWins"] / pairStats["matchesPlayed"]
                h2hDrawRate = pairStats["draws"] / pairStats["matchesPlayed"]
            else:
                homeH2HWinRate = np.nan
                awayH2HWinRate = np.nan
                h2hDrawRate = np.nan

            tournamentCategory = categorizeTournament(tournament)
            target = row.matchResult

            featureRows.append(
                {
                    "date": matchDate,
                    "homeTeam": homeTeam,
                    "awayTeam": awayTeam,
                    "tournament": tournament,
                    "tournamentCategory": tournamentCategory,
                    "neutral": neutralFlag,
                    "matchMonth": matchDate.month,
                    "matchYear": matchDate.year,
                    "homeScore": homeScore,
                    "awayScore": awayScore,
                    "homeElo": homeElo,
                    "awayElo": awayElo,
                    "eloDiff": eloDiff,
                    "absEloDiff": absEloDiff,
                    "eloExpectedHomeWin": eloExpectedHomeWin,
                    "homeMatchesPlayed": homeMatchesPlayed,
                    "awayMatchesPlayed": awayMatchesPlayed,
                    "matchesPlayedDiff": homeMatchesPlayed - awayMatchesPlayed,
                    "homeWinRate": homeWinRate,
                    "awayWinRate": awayWinRate,
                    "winRateDiff": homeWinRate - awayWinRate if pd.notna(homeWinRate) and pd.notna(awayWinRate) else np.nan,
                    "homeDrawRate": homeDrawRate,
                    "awayDrawRate": awayDrawRate,
                    "drawRateDiff": homeDrawRate - awayDrawRate if pd.notna(homeDrawRate) and pd.notna(awayDrawRate) else np.nan,
                    "homeLossRate": homeLossRate,
                    "awayLossRate": awayLossRate,
                    "lossRateDiff": homeLossRate - awayLossRate if pd.notna(homeLossRate) and pd.notna(awayLossRate) else np.nan,
                    "homeGoalsForAvg": homeGoalsForAvg,
                    "awayGoalsForAvg": awayGoalsForAvg,
                    "goalsForAvgDiff": homeGoalsForAvg - awayGoalsForAvg if pd.notna(homeGoalsForAvg) and pd.notna(awayGoalsForAvg) else np.nan,
                    "homeGoalsAgainstAvg": homeGoalsAgainstAvg,
                    "awayGoalsAgainstAvg": awayGoalsAgainstAvg,
                    "goalsAgainstAvgDiff": homeGoalsAgainstAvg - awayGoalsAgainstAvg if pd.notna(homeGoalsAgainstAvg) and pd.notna(awayGoalsAgainstAvg) else np.nan,
                    "homeGoalDiffAvg": homeGoalDiffAvg,
                    "awayGoalDiffAvg": awayGoalDiffAvg,
                    "goalDiffAvgDiff": homeGoalDiffAvg - awayGoalDiffAvg if pd.notna(homeGoalDiffAvg) and pd.notna(awayGoalDiffAvg) else np.nan,
                    "homeLast5WinRate": homeLast5WinRate,
                    "awayLast5WinRate": awayLast5WinRate,
                    "last5WinRateDiff": homeLast5WinRate - awayLast5WinRate if pd.notna(homeLast5WinRate) and pd.notna(awayLast5WinRate) else np.nan,
                    "homeLast5PointsAvg": homeLast5PointsAvg,
                    "awayLast5PointsAvg": awayLast5PointsAvg,
                    "last5PointsDiff": homeLast5PointsAvg - awayLast5PointsAvg if pd.notna(homeLast5PointsAvg) and pd.notna(awayLast5PointsAvg) else np.nan,
                    "homeLast5GoalsForAvg": homeLast5GoalsForAvg,
                    "awayLast5GoalsForAvg": awayLast5GoalsForAvg,
                    "last5GoalsForDiff": homeLast5GoalsForAvg - awayLast5GoalsForAvg if pd.notna(homeLast5GoalsForAvg) and pd.notna(awayLast5GoalsForAvg) else np.nan,
                    "homeLast5GoalsAgainstAvg": homeLast5GoalsAgainstAvg,
                    "awayLast5GoalsAgainstAvg": awayLast5GoalsAgainstAvg,
                    "last5GoalsAgainstDiff": homeLast5GoalsAgainstAvg - awayLast5GoalsAgainstAvg if pd.notna(homeLast5GoalsAgainstAvg) and pd.notna(awayLast5GoalsAgainstAvg) else np.nan,
                    "homeLast5GoalDiffAvg": homeLast5GoalDiffAvg,
                    "awayLast5GoalDiffAvg": awayLast5GoalDiffAvg,
                    "last5GoalDiffDiff": homeLast5GoalDiffAvg - awayLast5GoalDiffAvg if pd.notna(homeLast5GoalDiffAvg) and pd.notna(awayLast5GoalDiffAvg) else np.nan,
                    "homeLast10PointsAvg": homeLast10PointsAvg,
                    "awayLast10PointsAvg": awayLast10PointsAvg,
                    "last10PointsDiff": homeLast10PointsAvg - awayLast10PointsAvg if pd.notna(homeLast10PointsAvg) and pd.notna(awayLast10PointsAvg) else np.nan,
                    "homeLast10GoalsForAvg": homeLast10GoalsForAvg,
                    "awayLast10GoalsForAvg": awayLast10GoalsForAvg,
                    "last10GoalsForDiff": homeLast10GoalsForAvg - awayLast10GoalsForAvg if pd.notna(homeLast10GoalsForAvg) and pd.notna(awayLast10GoalsForAvg) else np.nan,
                    "homeLast10GoalsAgainstAvg": homeLast10GoalsAgainstAvg,
                    "awayLast10GoalsAgainstAvg": awayLast10GoalsAgainstAvg,
                    "last10GoalsAgainstDiff": homeLast10GoalsAgainstAvg - awayLast10GoalsAgainstAvg if pd.notna(homeLast10GoalsAgainstAvg) and pd.notna(awayLast10GoalsAgainstAvg) else np.nan,
                    "homeDaysSinceLastMatch": homeDaysSinceLastMatch,
                    "awayDaysSinceLastMatch": awayDaysSinceLastMatch,
                    "daysSinceLastMatchDiff": homeDaysSinceLastMatch - awayDaysSinceLastMatch if pd.notna(homeDaysSinceLastMatch) and pd.notna(awayDaysSinceLastMatch) else np.nan,
                    "homeH2HWinRate": homeH2HWinRate,
                    "awayH2HWinRate": awayH2HWinRate,
                    "h2hDrawRate": h2hDrawRate,
                    "target": target,
                }
            )

            homePoints, awayPoints = getResultPoints(homeScore, awayScore)

            homeState["matchesPlayed"] += 1
            awayState["matchesPlayed"] += 1
            homeState["wins"] += int(homeScore > awayScore)
            homeState["draws"] += int(homeScore == awayScore)
            homeState["losses"] += int(homeScore < awayScore)
            awayState["wins"] += int(awayScore > homeScore)
            awayState["draws"] += int(homeScore == awayScore)
            awayState["losses"] += int(awayScore < homeScore)

            homeState["goalsFor"] += homeScore
            homeState["goalsAgainst"] += awayScore
            awayState["goalsFor"] += awayScore
            awayState["goalsAgainst"] += homeScore

            homeState["last5Points"].append(homePoints)
            awayState["last5Points"].append(awayPoints)
            homeState["last5GoalsFor"].append(homeScore)
            homeState["last5GoalsAgainst"].append(awayScore)
            awayState["last5GoalsFor"].append(awayScore)
            awayState["last5GoalsAgainst"].append(homeScore)

            homeState["last10Points"].append(homePoints)
            awayState["last10Points"].append(awayPoints)
            homeState["last10GoalsFor"].append(homeScore)
            homeState["last10GoalsAgainst"].append(awayScore)
            awayState["last10GoalsFor"].append(awayScore)
            awayState["last10GoalsAgainst"].append(homeScore)

            homeState["lastMatchDate"] = matchDate
            awayState["lastMatchDate"] = matchDate

            newHomeElo, newAwayElo, _ = updateElo(homeElo, awayElo, homeScore, awayScore, neutral=bool(neutralFlag))
            homeState["elo"] = newHomeElo
            awayState["elo"] = newAwayElo

            pairStats["matchesPlayed"] += 1
            if homeScore > awayScore:
                if homeTeam == pairKey[0]:
                    pairStats["firstTeamWins"] += 1
                else:
                    pairStats["secondTeamWins"] += 1
            elif homeScore < awayScore:
                if awayTeam == pairKey[0]:
                    pairStats["firstTeamWins"] += 1
                else:
                    pairStats["secondTeamWins"] += 1
            else:
                pairStats["draws"] += 1

        modelDf = pd.DataFrame(featureRows)
        modelDf = modelDf.sort_values("date", kind="mergesort").reset_index(drop=True)

        print("Shape modelDf:", modelDf.shape)
        modelDf.head()
        """
    ),
    code(
        """
        nullSummary = modelDf.isnull().mean().sort_values(ascending=False)
        nullSummary[nullSummary > 0].head(20)
        """
    ),
    md("## 5. Imputacion controlada y target encoding"),
    code(
        """
        numericColumns = modelDf.select_dtypes(include=[np.number]).columns.tolist()
        for col in numericColumns:
            if modelDf[col].isnull().any():
                modelDf[col] = modelDf[col].fillna(modelDf[col].median())

        targetMapping = {"loss": 0, "draw": 1, "win": 2}
        modelDf["targetEncoded"] = modelDf["target"].map(targetMapping)

        modelDf = pd.get_dummies(
            modelDf,
            columns=["tournamentCategory"],
            prefix="tournamentCat",
            dtype=int,
        )

        remainingNulls = modelDf.isnull().sum().sum()
        assert remainingNulls == 0, f"Quedaron nulos en modelDf: {remainingNulls}"

        idColumns = [
            "date",
            "homeTeam",
            "awayTeam",
            "tournament",
            "neutral",
            "matchMonth",
            "matchYear",
            "homeScore",
            "awayScore",
            "target",
            "targetEncoded",
        ]
        otherColumns = [col for col in modelDf.columns if col not in idColumns]
        modelDf = modelDf[idColumns + otherColumns]

        print("Shape modelDf luego de imputacion y encoding:", modelDf.shape)
        modelDf.head()
        """
    ),
    md("## 6. Exportacion de datasets"),
    code(
        """
        finalModelDf = modelDf.drop(
            columns=[
                "homeTeam",
                "awayTeam",
                "tournament",
                "homeScore",
                "awayScore",
            ]
        ).copy()

        featureColumns = [
            col
            for col in modelDf.columns
            if col
            not in [
                "date",
                "homeTeam",
                "awayTeam",
                "tournament",
                "homeScore",
                "awayScore",
                "target",
                "targetEncoded",
            ]
        ]

        modelingBasePath.parent.mkdir(parents=True, exist_ok=True)
        finalModelPath.parent.mkdir(parents=True, exist_ok=True)

        modelDf.to_csv(modelingBasePath, index=False)
        finalModelDf.to_csv(finalModelPath, index=False)

        print("Numero de features utiles:", len(featureColumns))
        print("Shape modeling base:", modelDf.shape)
        print("Shape final model df:", finalModelDf.shape)
        print("Exportaciones completadas.")
        """
    ),
    code(
        """
        reloadedBaseDf = pd.read_csv(modelingBasePath, nrows=3)
        reloadedFinalDf = pd.read_csv(finalModelPath, nrows=3)

        print("Columnas modeling base:", len(reloadedBaseDf.columns))
        print("Columnas final model df:", len(reloadedFinalDf.columns))
        reloadedBaseDf.head()
        """
    ),
]


nb03_cells = [
    md(
        """
        # 03. Modelos Base y Evaluacion

        Este notebook toma el dataset maestro generado en Notebook 02 y construye:
        - baseline multiclase
        - modelos base de clasificacion
        - comparacion probabilistica por log loss
        - modelo Poisson para distribucion de goles
        """
    ),
    code(
        """
        import os
        from pathlib import Path

        os.environ["LOKY_MAX_CPU_COUNT"] = "1"

        import numpy as np
        import pandas as pd
        from scipy.stats import poisson
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression, PoissonRegressor
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            log_loss,
        )
        from sklearn.preprocessing import StandardScaler

        pd.set_option("display.max_columns", None)

        modelingBasePath = Path("../data/intermediate/modeling_base_dataset.csv")
        print("Modeling base path:", modelingBasePath)
        """
    ),
    md("## 1. Carga del dataset maestro"),
    code(
        """
        df = pd.read_csv(modelingBasePath)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

        print("Shape df:", df.shape)
        df.head()
        """
    ),
    md("## 2. Validacion y definicion de features"),
    code(
        """
        requiredColumns = [
            "date",
            "homeTeam",
            "awayTeam",
            "tournament",
            "homeScore",
            "awayScore",
            "target",
            "targetEncoded",
        ]

        missingColumns = [col for col in requiredColumns if col not in df.columns]
        assert not missingColumns, f"Faltan columnas requeridas: {missingColumns}"
        assert df[requiredColumns].isnull().sum().sum() == 0, "Hay nulos en columnas criticas"

        nonFeatureColumns = [
            "date",
            "homeTeam",
            "awayTeam",
            "tournament",
            "homeScore",
            "awayScore",
            "target",
            "targetEncoded",
        ]
        featureColumns = [col for col in df.columns if col not in nonFeatureColumns]

        X = df[featureColumns].copy()
        y = df["targetEncoded"].copy()

        nonNumericFeatureColumns = X.select_dtypes(exclude=[np.number]).columns.tolist()
        assert not nonNumericFeatureColumns, f"Hay features no numericas: {nonNumericFeatureColumns}"

        print("Numero de features:", len(featureColumns))
        print("Distribucion target:")
        print(df["target"].value_counts(normalize=True).round(4))
        """
    ),
    md("## 3. Split temporal"),
    code(
        """
        trainEnd = int(len(df) * 0.70)
        valEnd = trainEnd + int(len(df) * 0.15)

        trainDf = df.iloc[:trainEnd].copy()
        valDf = df.iloc[trainEnd:valEnd].copy()
        testDf = df.iloc[valEnd:].copy()

        X_train = trainDf[featureColumns].copy()
        X_val = valDf[featureColumns].copy()
        X_test = testDf[featureColumns].copy()

        y_train = trainDf["targetEncoded"].copy()
        y_val = valDf["targetEncoded"].copy()
        y_test = testDf["targetEncoded"].copy()

        print("Train:", X_train.shape)
        print("Val:", X_val.shape)
        print("Test:", X_test.shape)
        """
    ),
    md("## 4. Baseline de referencia"),
    code(
        """
        majorityClass = y_train.value_counts().idxmax()
        y_val_pred_base = np.full_like(y_val, majorityClass)

        print("BASELINE (VAL)")
        print("Accuracy:", accuracy_score(y_val, y_val_pred_base))
        print("Balanced Accuracy:", balanced_accuracy_score(y_val, y_val_pred_base))
        print("F1:", f1_score(y_val, y_val_pred_base, average="macro"))
        """
    ),
    md("## 5. Modelos base de clasificacion"),
    code(
        """
        classifierScaler = StandardScaler()
        X_train_scaled = classifierScaler.fit_transform(X_train)
        X_val_scaled = classifierScaler.transform(X_val)
        X_test_scaled = classifierScaler.transform(X_test)

        logModel = LogisticRegression(max_iter=2000)
        rfModel = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
        )

        logModel.fit(X_train_scaled, y_train)
        rfModel.fit(X_train, y_train)


        def metricRow(modelName, yTrue, yPred, yProba):
            return {
                "model": modelName,
                "accuracy": accuracy_score(yTrue, yPred),
                "balancedAccuracy": balanced_accuracy_score(yTrue, yPred),
                "f1": f1_score(yTrue, yPred, average="macro"),
                "logLoss": log_loss(yTrue, yProba, labels=[0, 1, 2]),
            }


        y_val_pred_log = logModel.predict(X_val_scaled)
        y_val_proba_log = logModel.predict_proba(X_val_scaled)
        y_val_pred_rf = rfModel.predict(X_val)
        y_val_proba_rf = rfModel.predict_proba(X_val)

        validationResults = pd.DataFrame(
            [
                metricRow("LogisticRegression", y_val, y_val_pred_log, y_val_proba_log),
                metricRow("RandomForest", y_val, y_val_pred_rf, y_val_proba_rf),
            ]
        ).sort_values(["logLoss", "balancedAccuracy", "f1"], ascending=[True, False, False]).reset_index(drop=True)

        validationResults
        """
    ),
    md("## 6. Seleccion automatica del mejor clasificador"),
    code(
        """
        bestClassifierName = validationResults.iloc[0]["model"]

        classifierSpecs = {
            "LogisticRegression": {
                "model": logModel,
                "X_test": X_test_scaled,
            },
            "RandomForest": {
                "model": rfModel,
                "X_test": X_test,
            },
        }

        bestClassifier = classifierSpecs[bestClassifierName]["model"]
        bestClassifierXTest = classifierSpecs[bestClassifierName]["X_test"]

        y_test_pred_best = bestClassifier.predict(bestClassifierXTest)
        y_test_proba_best = bestClassifier.predict_proba(bestClassifierXTest)

        testBestResults = pd.DataFrame(
            [metricRow(bestClassifierName, y_test, y_test_pred_best, y_test_proba_best)]
        )

        print("Best classifier selected:", bestClassifierName)
        testBestResults
        """
    ),
    md("## 7. Evaluacion final del mejor clasificador"),
    code(
        """
        print("TEST RESULTS -", bestClassifierName)
        print("Accuracy:", accuracy_score(y_test, y_test_pred_best))
        print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred_best))
        print("F1:", f1_score(y_test, y_test_pred_best, average="macro"))
        print("LogLoss:", log_loss(y_test, y_test_proba_best, labels=[0, 1, 2]))
        print()
        print(classification_report(y_test, y_test_pred_best, zero_division=0))
        """
    ),
    code(
        """
        cm = confusion_matrix(y_test, y_test_pred_best)

        print("CONFUSION MATRIX -", bestClassifierName)
        print(cm)
        print()
        print("Rows = Actual | Columns = Predicted")
        print("Classes: [0=loss, 1=draw, 2=win]")
        """
    ),
    code(
        """
        if bestClassifierName == "RandomForest":
            importanceDf = (
                pd.DataFrame(
                    {
                        "feature": featureColumns,
                        "importance": bestClassifier.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=False)
                .head(15)
                .reset_index(drop=True)
            )
            importanceDf
        else:
            coefDf = (
                pd.DataFrame(logModel.coef_.T, index=featureColumns, columns=["coef_loss", "coef_draw", "coef_win"])
                .assign(absImpact=lambda frame: frame.abs().sum(axis=1))
                .sort_values("absImpact", ascending=False)
                .head(15)
                .reset_index(drop=False)
                .rename(columns={"index": "feature"})
            )
            coefDf
        """
    ),
    md("## 8. Poisson goal model"),
    code(
        """
        poissonFeatureColumns = featureColumns.copy()

        poissonScaler = StandardScaler()
        X_train_p = poissonScaler.fit_transform(trainDf[poissonFeatureColumns])
        X_val_p = poissonScaler.transform(valDf[poissonFeatureColumns])
        X_test_p = poissonScaler.transform(testDf[poissonFeatureColumns])

        y_home_train = trainDf["homeScore"].to_numpy()
        y_away_train = trainDf["awayScore"].to_numpy()

        poissonHome = PoissonRegressor(alpha=0.5, max_iter=1000)
        poissonAway = PoissonRegressor(alpha=0.5, max_iter=1000)

        poissonHome.fit(X_train_p, y_home_train)
        poissonAway.fit(X_train_p, trainDf["awayScore"].to_numpy())

        lambdaHome_val = np.clip(poissonHome.predict(X_val_p), 0.01, None)
        lambdaAway_val = np.clip(poissonAway.predict(X_val_p), 0.01, None)
        lambdaHome_test = np.clip(poissonHome.predict(X_test_p), 0.01, None)
        lambdaAway_test = np.clip(poissonAway.predict(X_test_p), 0.01, None)
        """
    ),
    code(
        """
        def matchOutcomeProbabilities(lambdaHome, lambdaAway, maxGoals=10):
            goals = np.arange(maxGoals + 1)
            pmfHome = poisson.pmf(goals, lambdaHome)
            pmfAway = poisson.pmf(goals, lambdaAway)
            probabilityGrid = np.outer(pmfHome, pmfAway)

            totalProbability = probabilityGrid.sum()
            if totalProbability > 0:
                probabilityGrid = probabilityGrid / totalProbability

            probWin = np.tril(probabilityGrid, -1).sum()
            probDraw = np.trace(probabilityGrid)
            probLoss = np.triu(probabilityGrid, 1).sum()

            return np.array([probLoss, probDraw, probWin])


        def predictPoissonResults(lambdaHomeArray, lambdaAwayArray, maxGoals=10):
            predictedClasses = []
            predictedProbabilities = []

            for lambdaHome, lambdaAway in zip(lambdaHomeArray, lambdaAwayArray):
                probs = matchOutcomeProbabilities(lambdaHome, lambdaAway, maxGoals=maxGoals)
                predictedClasses.append(int(np.argmax(probs)))
                predictedProbabilities.append(probs)

            return np.array(predictedClasses), np.array(predictedProbabilities)


        y_val_pred_poisson, y_val_proba_poisson = predictPoissonResults(lambdaHome_val, lambdaAway_val, maxGoals=10)
        y_test_pred_poisson, y_test_proba_poisson = predictPoissonResults(lambdaHome_test, lambdaAway_test, maxGoals=10)

        poissonValidationResults = pd.DataFrame(
            [metricRow("PoissonGoalModel", y_val, y_val_pred_poisson, y_val_proba_poisson)]
        )
        poissonTestResults = pd.DataFrame(
            [metricRow("PoissonGoalModel", y_test, y_test_pred_poisson, y_test_proba_poisson)]
        )

        poissonValidationResults
        """
    ),
    code(
        """
        print("POISSON MODEL (TEST)")
        print("Accuracy:", accuracy_score(y_test, y_test_pred_poisson))
        print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred_poisson))
        print("F1:", f1_score(y_test, y_test_pred_poisson, average="macro"))
        print("LogLoss:", log_loss(y_test, y_test_proba_poisson, labels=[0, 1, 2]))
        print()
        print(classification_report(y_test, y_test_pred_poisson, zero_division=0))
        """
    ),
    code(
        """
        comparisonDf = pd.concat(
            [
                validationResults.assign(split="validation"),
                poissonValidationResults.assign(split="validation"),
                testBestResults.assign(split="test"),
                poissonTestResults.assign(split="test"),
            ],
            ignore_index=True,
        )

        comparisonDf.sort_values(["split", "logLoss", "balancedAccuracy"], ascending=[True, True, False]).reset_index(drop=True)
        """
    ),
]


write_notebook(NOTEBOOKS_DIR / "01_data_foundation.ipynb", nb01_cells)
write_notebook(NOTEBOOKS_DIR / "02_feature_engineering.ipynb", nb02_cells)
write_notebook(NOTEBOOKS_DIR / "02_feature_engineering_CORRECTED.ipynb", nb02_cells)
write_notebook(NOTEBOOKS_DIR / "03_modelos_base_evaluacion.ipynb", nb03_cells)

print("Notebooks rebuilt successfully.")
