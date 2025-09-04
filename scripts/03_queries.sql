-- ============================================
-- SIMPLE QUERIES (Filtering)
-- ============================================

-- 1. Top-rated movies (rating > 8.0)
SELECT movie_id, title, vote_average, vote_count, release_year
FROM movies
WHERE vote_average > 8.0
ORDER BY vote_average DESC, vote_count DESC
LIMIT 20;

-- 2. Movies from a specific year
SELECT movie_id, title, vote_average, budget, revenue
FROM movies
WHERE release_year = 2020
ORDER BY vote_average DESC
LIMIT 15;

-- 3. Movies with large budgets (> 100 million)
SELECT movie_id, title, budget, revenue, (revenue - budget) as profit
FROM movies
WHERE budget > 100000000
ORDER BY budget DESC
LIMIT 10;

-- ============================================
-- COMPLEX QUERIES (Joins and combinations)
-- ============================================

-- 4. Movies with their categories from the key-value table
SELECT m.movie_id,
       m.title,
       m.vote_average,
       m.budget,
       kv_rating.key_value as rating_category,
       kv_budget.key_value as budget_category
FROM movies m
         LEFT JOIN movie_keyvalue kv_rating ON m.movie_id = kv_rating.movie_id
    AND kv_rating.key_name = 'rating_category'
         LEFT JOIN movie_keyvalue kv_budget ON m.movie_id = kv_budget.movie_id
    AND kv_budget.key_name = 'budget_category'
WHERE m.vote_average IS NOT NULL
ORDER BY m.vote_average DESC
LIMIT 20;

-- 5. Movies with runtime categories and profitability
SELECT m.movie_id,
       m.title,
       m.runtime,
       m.budget,
       m.revenue,
       kv_runtime.key_value as runtime_category,
       CASE
           WHEN m.revenue > m.budget * 2 THEN 'highly_profitable'
           WHEN m.revenue > m.budget THEN 'profitable'
           ELSE 'not_profitable'
           END              as profitability
FROM movies m
         LEFT JOIN movie_keyvalue kv_runtime ON m.movie_id = kv_runtime.movie_id
    AND kv_runtime.key_name = 'runtime_category'
WHERE m.budget > 0
  AND m.revenue > 0
ORDER BY (m.revenue - m.budget) DESC
LIMIT 25;

-- 6. Key-value analysis by categories
SELECT kv1.key_value       as rating_category,
       kv2.key_value       as budget_category,
       COUNT(*)            as movie_count,
       AVG(m.vote_average) as avg_rating,
       AVG(m.budget)       as avg_budget
FROM movie_keyvalue kv1
         JOIN movie_keyvalue kv2 ON kv1.movie_id = kv2.movie_id
         JOIN movies m ON kv1.movie_id = m.movie_id
WHERE kv1.key_name = 'rating_category'
  AND kv2.key_name = 'budget_category'
  AND m.budget > 0
GROUP BY kv1.key_value, kv2.key_value
ORDER BY movie_count DESC;

-- ============================================
-- COMPLEX QUERIES (Aggregated reports)
-- ============================================

-- 7. Detailed yearly report with aggregations
SELECT release_year,
       COUNT(*)                                       as total_movies,
       AVG(vote_average)                              as avg_rating,
       AVG(budget)                                    as avg_budget,
       AVG(revenue)                                   as avg_revenue,
       SUM(revenue)                                   as total_revenue,
       AVG(runtime)                                   as avg_runtime,
       COUNT(CASE WHEN vote_average > 7.0 THEN 1 END) as high_rated_movies,
       ROUND(AVG(revenue - budget), 2)                as avg_profit
FROM movies
WHERE release_year IS NOT NULL
  AND release_year BETWEEN 2010 AND 2023
  AND budget > 0
  AND revenue > 0
GROUP BY release_year
ORDER BY release_year DESC;

-- 8. Top production companies with performance metrics
WITH company_stats AS (
    SELECT TRIM(unnest(string_to_array(production_companies, ','))) as company,
           COUNT(*)                                                 as movie_count,
           AVG(vote_average)                                        as avg_rating,
           SUM(revenue)                                             as total_revenue,
           AVG(budget)                                              as avg_budget,
           SUM(revenue - budget)                                    as total_profit
    FROM movies
    WHERE production_companies IS NOT NULL
      AND production_companies != ''
      AND budget > 0
      AND revenue > 0
      AND release_year >= 2010
    GROUP BY company
)
SELECT company,
       movie_count,
       ROUND(avg_rating, 2)                 as avg_rating,
       total_revenue,
       ROUND(avg_budget, 2)                 as avg_budget,
       total_profit,
       ROUND(total_profit / movie_count, 2) as avg_profit_per_movie
FROM company_stats
WHERE movie_count >= 5
ORDER BY total_revenue DESC
LIMIT 15;

-- 9. Key-value aggregate report by genre
WITH genre_analysis AS (
    SELECT TRIM(unnest(string_to_array(m.genres, ','))) as genre,
           m.movie_id,
           m.vote_average,
           m.budget,
           m.revenue,
           kv.key_value                                 as rating_category
    FROM movies m
         LEFT JOIN movie_keyvalue kv ON m.movie_id = kv.movie_id
        AND kv.key_name = 'rating_category'
    WHERE m.genres IS NOT NULL
      AND m.genres != ''
      AND m.release_year >= 2000
)
SELECT genre,
       COUNT(*)                                                  as total_movies,
       ROUND(AVG(vote_average), 2)                               as avg_rating,
       COUNT(CASE WHEN rating_category = 'excellent' THEN 1 END) as excellent_movies,
       COUNT(CASE WHEN rating_category = 'good' THEN 1 END)      as good_movies,
       ROUND(AVG(budget), 2)                                     as avg_budget,
       ROUND(AVG(revenue), 2)                                    as avg_revenue,
       ROUND(SUM(revenue - budget), 2)                           as total_profit
FROM genre_analysis
WHERE budget > 0
  AND revenue > 0
GROUP BY genre
HAVING COUNT(*) >= 10
ORDER BY avg_rating DESC, total_movies DESC
LIMIT 20;

-- 10. Time series analysis of the movie industry with key-value metrics
SELECT m.release_year,
       COUNT(*)                                                                      as total_movies,

       -- Rating analysis
       ROUND(AVG(m.vote_average), 2)                                                 as avg_rating,
       COUNT(CASE WHEN kv_rating.key_value = 'excellent' THEN 1 END)                 as excellent_count,
       COUNT(CASE WHEN kv_rating.key_value = 'good' THEN 1 END)                      as good_count,

       -- Budget analysis
       ROUND(AVG(m.budget), 2)                                                       as avg_budget,
       COUNT(CASE WHEN kv_budget.key_value = 'blockbuster' THEN 1 END)               as blockbuster_count,
       COUNT(CASE WHEN kv_budget.key_value = 'big_budget' THEN 1 END)                as big_budget_count,

       -- Runtime analysis
       ROUND(AVG(m.runtime), 2)                                                      as avg_runtime,
       COUNT(CASE WHEN kv_runtime.key_value = 'long' THEN 1 END)                     as long_movies,
       COUNT(CASE WHEN kv_runtime.key_value = 'very_long' THEN 1 END)                as very_long_movies,

       -- Financial analysis
       ROUND(SUM(m.revenue - m.budget) / 1000000, 2)                                 as total_profit_millions,
       ROUND(AVG(CASE WHEN m.revenue > 0 THEN (m.revenue::float / m.budget) END), 2) as avg_roi

FROM movies m
         LEFT JOIN movie_keyvalue kv_rating ON m.movie_id = kv_rating.movie_id
    AND kv_rating.key_name = 'rating_category'
         LEFT JOIN movie_keyvalue kv_budget ON m.movie_id = kv_budget.movie_id
    AND kv_budget.key_name = 'budget_category'
         LEFT JOIN movie_keyvalue kv_runtime ON m.movie_id = kv_runtime.movie_id
    AND kv_runtime.key_name = 'runtime_category'
WHERE m.release_year BETWEEN 2000 AND 2023
  AND m.budget > 0
  AND m.revenue > 0
GROUP BY m.release_year
ORDER BY m.release_year;

-- ============================================
-- PERFORMANCE TESTING QUERIES
-- ============================================

-- Performance test - Key-value approach
EXPLAIN ANALYZE
SELECT m.title, kv.key_name, kv.key_value
FROM movies m
         JOIN movie_keyvalue kv ON m.movie_id = kv.movie_id
WHERE kv.key_name IN ('rating_category', 'budget_category')
  AND m.release_year = 2020;

-- Performance test - Traditional approach
EXPLAIN ANALYZE
SELECT title,
       CASE
           WHEN vote_average >= 8.0 THEN 'excellent'
           WHEN vote_average >= 7.0 THEN 'good'
           ELSE 'average'
           END as rating_category,
       CASE
           WHEN budget >= 100000000 THEN 'blockbuster'
           WHEN budget >= 50000000 THEN 'big_budget'
           ELSE 'low_budget'
           END as budget_category
FROM movies
WHERE release_year = 2020;
