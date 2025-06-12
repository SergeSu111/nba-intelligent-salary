import sqlite3
import pandas as pd


def run_queries(db_path='nba_salary_project.db'):
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query('SELECT * FROM PlayerStatsOnCourt LIMIT 10', conn)
    print(df)

    # Query 1: Average Salary by Age Group
    df = pd.read_sql_query('''
    SELECT Age, AVG(salary) AS avg_salary
    FROM PlayerStats
    GROUP BY Age
    ORDER BY Age;''', conn)
    print("\nQuery 1: Average Salary by Age Group")
    print(df)

    # Query 2: Correlation Between Salary and Points Per Game (PPG)
    df = pd.read_sql_query('''SELECT PPG, AVG(salary) AS avg_salary
    FROM PlayerStats
    GROUP BY PPG
    ORDER BY PPG DESC;''', conn)
    print("\nQuery 2: Correlation Between Salary and Points Per Game (PPG)")
    print(df)

    # Query 3: Top 10 Players with the Highest Salary and their Shooting Efficiency
    df = pd.read_sql_query('''SELECT Player, salary, ShootingEfficiency
    FROM PlayerStats
    ORDER BY salary DESC
    LIMIT 10
    ''', conn)
    print("\nQuery 3: Top 10 Players with the Highest Salary and their Shooting Efficiency")
    print(df)

    # Query 4: Salary Comparison to WEFF (Weighted Efficiency)
    df = pd.read_sql_query('''
    SELECT WEFF, AVG(salary) AS avg_salary
    FROM PlayerStats
    GROUP BY WEFF
    ORDER BY WEFF DESC;
    ''', conn)
    print("\nQuery 4: Salary Comparison to WEFF (Weighted Efficiency)")
    print(df)

    # Query 5: Impact of Games Started (GS%) on Salary
    df = pd.read_sql_query('''
    SELECT "GS%", AVG(salary) AS avg_salary
    FROM PlayerStats
    GROUP BY "GS%"
    ORDER BY "GS%" DESC;''', conn)
    print("\nQuery 5: Impact of Games Started (GS%) on Salary")
    print(df)

    # Query 6: Relationship Between Minutes Per Game (MPG) and Salary
    df = pd.read_sql_query('''
    SELECT MPG, AVG(salary) AS avg_salary
    FROM PlayerStats
    GROUP BY MPG
    ORDER BY MPG DESC;''', conn)
    print("\nQuery 6: Relationship Between Minutes Per Game (MPG) and Salary")
    print(df)

if __name__ == '__main__':
    run_queries()