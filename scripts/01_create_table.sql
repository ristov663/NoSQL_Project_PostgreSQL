-- Create the main table for movies
CREATE TABLE IF NOT EXISTS movies
(
    id                      SERIAL PRIMARY KEY,
    movie_id                INTEGER UNIQUE,
    title                   TEXT,
    vote_average            DECIMAL(3, 1),
    vote_count              INTEGER,
    status                  TEXT,
    release_date            DATE,
    revenue                 BIGINT,
    runtime                 INTEGER,
    adult                   BOOLEAN,
    genres                  TEXT,
    overview_sentiment      TEXT,
    cast_members            TEXT,
    crew_members            TEXT,
    genres_list             TEXT,
    keywords                TEXT,
    director_of_photography TEXT,
    producers               TEXT,
    music_composer          TEXT,
    star1                   TEXT,
    star2                   TEXT,
    star3                   TEXT,
    star4                   TEXT,
    writer                  TEXT,
    original_language       TEXT,
    original_title          TEXT,
    popularity              DECIMAL(10, 3),
    budget                  BIGINT,
    tagline                 TEXT,
    production_companies    TEXT,
    production_countries    TEXT,
    spoken_languages        TEXT,
    homepage                TEXT,
    imdb_id                 TEXT,
    tmdb_id                 TEXT,
    video                   BOOLEAN,
    poster_path             TEXT,
    backdrop_path           TEXT,
    release_year            INTEGER,
    collection_name         TEXT,
    collection_id           TEXT,
    genres_id               TEXT,
    original_language_code  TEXT,
    overview                TEXT,
    all_combined_keywords   TEXT
);

-- Create indexes for better performance
CREATE INDEX idx_movies_title ON movies (title);
CREATE INDEX idx_movies_release_year ON movies (release_year);
CREATE INDEX idx_movies_vote_average ON movies (vote_average);
CREATE INDEX idx_movies_revenue ON movies (revenue);
CREATE INDEX idx_movies_budget ON movies (budget);
CREATE INDEX idx_movies_popularity ON movies (popularity);
CREATE INDEX idx_movies_runtime ON movies (runtime);

-- Key-value table for flexible storage
CREATE TABLE IF NOT EXISTS movie_keyvalue
(
    id         SERIAL PRIMARY KEY,
    movie_id   INTEGER REFERENCES movies (movie_id),
    key_name   TEXT NOT NULL,
    key_value  TEXT,
    key_type   TEXT      DEFAULT 'string',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_keyvalue_movie_id ON movie_keyvalue (movie_id);
CREATE INDEX idx_keyvalue_key_name ON movie_keyvalue (key_name);
CREATE INDEX idx_keyvalue_key_value ON movie_keyvalue (key_value);
