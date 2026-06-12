
import soccerdata as sd


def test_sofifa():
    try:
        # SoFIFA covers all international teams.
        sofifa = sd.SoFIFA(leagues="FIFA World Cup")
        df = sofifa.read_team_players()
        print("SoFIFA Data:", df.head())
    except Exception as e:
        print("SoFIFA Error:", e)


def test_fbref():
    try:
        fbref = sd.FBref(leagues="FIFA World Cup", seasons=2022)
        df = fbref.read_match_stats(stat_type="possession")
        print("FBref Data:", df.head())
    except Exception as e:
        print("FBref Error:", e)


if __name__ == "__main__":
    test_sofifa()
    test_fbref()
