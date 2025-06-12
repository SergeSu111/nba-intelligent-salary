import sqlite3
import pandas as pd

def init_db(db_path = '../db/nba_salary_project.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the PlayerStats table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS PlayerStatsOnCourt (
        id INTEGER PRIMARY KEY,
        Season INTEGER,
        Player TEXT,
        Pos TEXT,
        Age INTEGER,
        Team TEXT,
        G INTEGER,
        GS INTEGER,
        MP INTEGER,
        FG INTEGER,
        FGA INTEGER,
        FG_perc REAL,
        ThreeP INTEGER,
        ThreePA INTEGER,
        ThreeP_perc REAL,
        TwoP INTEGER,
        TwoPA INTEGER,
        TwoP_perc REAL,
        eFG_perc REAL,
        FT INTEGER,
        FTA INTEGER,
        FT_perc REAL,
        ORB INTEGER,
        DRB INTEGER,
        TRB INTEGER,
        AST INTEGER,
        STL INTEGER,
        BLK INTEGER,
        TOV INTEGER,
        PF INTEGER,
        PTS INTEGER,
        playerName TEXT,
        seasonStartYear INTEGER,
        salary REAL,
        WEFF REAL,
        PPG REAL,
        APG REAL,
        RPG REAL,
        SPG REAL,
        BPG REAL,
        TPG REAL,
        UsageRate REAL,
        ShootingEfficiency REAL,
        OffensiveContribution REAL,
        DefensiveContribution REAL,
        Experience INTEGER,
        GS_perc REAL,
        ImpactScore REAL,
        MPG REAL,
        WEFFRange TEXT,
        EfficiencyTier TEXT
    )
    ''')

    conn.commit()
    print("Table created successfully.")
    conn.close()

if __name__ == '__main__':
    init_db()