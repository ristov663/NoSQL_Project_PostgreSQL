import pandas as pd
import psycopg2
import numpy as np
import sys
import os

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'movie_db',
    'user': 'user',
    'password': 'password'
}


def create_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Connection error: {e}")
        return None


def clean_data(df):
    """Clean and preprocess the data before database import"""
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Replace NaN with empty strings
    df = df.fillna('')

    # Convert release_date column to datetime
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Convert boolean-like columns to True/False
    if 'adult' in df.columns:
        df['adult'] = df['adult'].astype(str).str.lower() == 'true'

    if 'video' in df.columns:
        df['video'] = df['video'].astype(str).str.lower() == 'true'

    # Truncate long text fields to max 1000 characters
    text_columns = ['title', 'original_title', 'tagline', 'overview']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str[:1000]

    return df


def import_movies(csv_file):
    """Import movie data from a CSV file into PostgreSQL"""
    conn = create_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()

        # Read CSV file into DataFrame
        print("Reading CSV file...")
        df = pd.read_csv(csv_file, encoding='utf-8')

        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")

        # Clean and preprocess the dataset
        df = clean_data(df)

        # Map input CSV column names to PostgreSQL table column names
        column_mapping = {
            'Unnamed: 0': 'movie_id',
            'title': 'title',
            'vote_average': 'vote_average',
            'vote_count': 'vote_count',
            'status': 'status',
            'release_date': 'release_date',
            'revenue': 'revenue',
            'runtime': 'runtime',
            'adult': 'adult',
            'genres': 'genres',
            'overview_sentiment': 'overview_sentiment',
            'cast': 'cast_members',
            'crew': 'crew_members',
            'genres_list': 'genres_list',
            'keywords': 'keywords',
            'Director of Photography': 'director_of_photography',
            'Producers': 'producers',
            'Music Composer': 'music_composer',
            'Star1': 'star1',
            'Star2': 'star2',
            'Star3': 'star3',
            'Star4': 'star4',
            'Writer': 'writer',
            'original_language': 'original_language',
            'original_title': 'original_title',
            'popularity': 'popularity',
            'budget': 'budget',
            'tagline': 'tagline',
            'production_companies': 'production_companies',
            'production_countries': 'production_countries',
            'spoken_languages': 'spoken_languages',
            'homepage': 'homepage',
            'imdb_id': 'imdb_id',
            'id': 'tmdb_id',
            'video': 'video',
            'poster_path': 'poster_path',
            'backdrop_path': 'backdrop_path',
            'Release Year': 'release_year',
            'Collection Name': 'collection_name',
            'Collection ID': 'collection_id',
            'Genres ID': 'genres_id',
            'Original Language Code': 'original_language_code',
            'overview': 'overview',
            'All Combined Keywords': 'all_combined_keywords'
        }

        # Rename columns according to the defined mapping
        df_renamed = df.rename(columns=column_mapping)

        # Define required columns that exist in the "movies" table
        required_columns = [
            'movie_id', 'title', 'vote_average', 'vote_count', 'status',
            'release_date', 'revenue', 'runtime', 'adult', 'genres',
            'overview_sentiment', 'cast_members', 'crew_members', 'genres_list',
            'keywords', 'director_of_photography', 'producers', 'music_composer',
            'star1', 'star2', 'star3', 'star4', 'writer', 'original_language',
            'original_title', 'popularity', 'budget', 'tagline',
            'production_companies', 'production_countries', 'spoken_languages',
            'homepage', 'imdb_id', 'tmdb_id', 'video', 'poster_path',
            'backdrop_path', 'release_year', 'collection_name', 'collection_id',
            'genres_id', 'original_language_code', 'overview', 'all_combined_keywords'
        ]

        # Keep only existing/available columns from CSV
        existing_columns = [col for col in required_columns if col in df_renamed.columns]
        df_final = df_renamed[existing_columns]

        print(f"Using {len(existing_columns)} columns")

        # Insert data in batches for performance
        batch_size = 1000
        total_rows = len(df_final)

        for i in range(0, total_rows, batch_size):
            batch_df = df_final.iloc[i:i + batch_size]

            values = []
            for _, row in batch_df.iterrows():
                row_values = []
                for col in existing_columns:
                    value = row[col]
                    if pd.isna(value) or value == '':
                        row_values.append('NULL')
                    elif col in ['adult', 'video']:
                        # Store booleans as TRUE/FALSE
                        row_values.append(str(bool(value)).upper())
                    elif col == 'release_date':
                        # Format dates in SQL-compatible format
                        if pd.isna(value):
                            row_values.append('NULL')
                        else:
                            row_values.append(f"'{value.strftime('%Y-%m-%d')}'")
                    elif isinstance(value, str):
                        # Escape single quotes for SQL safety
                        escaped_value = value.replace("'", "''")
                        row_values.append(f"'{escaped_value}'")
                    else:
                        row_values.append(str(value))

                values.append(f"({', '.join(row_values)})")

            # Insert batch into DB
            if values:
                insert_query = f"""
                INSERT INTO movies ({', '.join(existing_columns)})
                VALUES {', '.join(values)}
                ON CONFLICT (movie_id) DO NOTHING;
                """

                cursor.execute(insert_query)
                conn.commit()

                print(f"Imported {i + len(batch_df)}/{total_rows} rows")

        print("Movie import completed successfully!")

        # Populate key-value table with derived attributes
        print("Populating key-value table...")
        populate_keyvalue_table(cursor, conn)

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"Error during import: {e}")
        conn.rollback()
        return False


def populate_keyvalue_table(cursor, conn):
    """Populate the key-value table with derived movie categories"""

    # Insert rating categories (excellent, good, etc.)
    cursor.execute("""
        INSERT INTO movie_keyvalue (movie_id, key_name, key_value, key_type)
        SELECT 
            movie_id,
            'rating_category',
            CASE 
                WHEN vote_average >= 8.0 THEN 'excellent'
                WHEN vote_average >= 7.0 THEN 'good'
                WHEN vote_average >= 6.0 THEN 'average'
                WHEN vote_average >= 5.0 THEN 'below_average'
                ELSE 'poor'
            END,
            'string'
        FROM movies 
        WHERE vote_average IS NOT NULL;
    """)

    # Insert runtime categories (short, long, very_long, etc.)
    cursor.execute("""
        INSERT INTO movie_keyvalue (movie_id, key_name, key_value, key_type)
        SELECT 
            movie_id,
            'runtime_category',
            CASE 
                WHEN runtime >= 180 THEN 'very_long'
                WHEN runtime >= 120 THEN 'long'
                WHEN runtime >= 90 THEN 'standard'
                WHEN runtime >= 60 THEN 'short'
                ELSE 'very_short'
            END,
            'string'
        FROM movies 
        WHERE runtime IS NOT NULL;
    """)

    # Insert budget categories (blockbuster, low_budget, etc.)
    cursor.execute("""
        INSERT INTO movie_keyvalue (movie_id, key_name, key_value, key_type)
        SELECT 
            movie_id,
            'budget_category',
            CASE 
                WHEN budget >= 100000000 THEN 'blockbuster'
                WHEN budget >= 50000000 THEN 'big_budget'
                WHEN budget >= 10000000 THEN 'medium_budget'
                WHEN budget >= 1000000 THEN 'low_budget'
                ELSE 'micro_budget'
            END,
            'string'
        FROM movies 
        WHERE budget > 0;
    """)

    conn.commit()
    print("Key-value table populated!")


if __name__ == "__main__":
    csv_file = "../../NoSQL_Project_PostgreSQL/data/movie_dataset.csv"

    if not os.path.exists(csv_file):
        print(f"File {csv_file} does not exist!")
        sys.exit(1)

    success = import_movies(csv_file)
    if success:
        print("Movie import completed successfully!")
    else:
        print("Movie import failed!")
