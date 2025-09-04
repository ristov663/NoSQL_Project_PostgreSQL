import psycopg2
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Connection configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'movie_db',
    'user': 'user',
    'password': 'password'
}


def create_connection():
    """Create a connection to PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Connection error: {e}")
        return None


def execute_query_with_timing(cursor, query, description):
    """Execute query and measure execution time"""
    print(f"\nExecuting: {description}")
    print("=" * 50)

    start_time = time.time()
    cursor.execute(query)
    results = cursor.fetchall()
    end_time = time.time()

    execution_time = end_time - start_time
    row_count = len(results)

    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Number of results: {row_count}")

    return execution_time, row_count, results


def run_performance_tests():
    """Run performance tests on different queries"""
    conn = create_connection()
    if not conn:
        return

    cursor = conn.cursor()
    results = []

    # Define the tests
    tests = [
        {
            'name': 'Simple Filtering - Top Movies',
            'query': """
                SELECT movie_id, title, vote_average, vote_count 
                FROM movies 
                WHERE vote_average > 8.0 
                ORDER BY vote_average DESC 
                LIMIT 100;
            """
        },
        {
            'name': 'Key-Value Join - Categories',
            'query': """
                SELECT m.movie_id, m.title, m.vote_average,
                       kv_rating.key_value as rating_category,
                       kv_budget.key_value as budget_category
                FROM movies m
                LEFT JOIN movie_keyvalue kv_rating ON m.movie_id = kv_rating.movie_id 
                    AND kv_rating.key_name = 'rating_category'
                LEFT JOIN movie_keyvalue kv_budget ON m.movie_id = kv_budget.movie_id 
                    AND kv_budget.key_name = 'budget_category'
                WHERE m.vote_average > 7.0
                ORDER BY m.vote_average DESC
                LIMIT 100;
            """
        },
        {
            'name': 'Yearly Aggregation',
            'query': """
                SELECT release_year,
                       COUNT(*) as total_movies,
                       AVG(vote_average) as avg_rating,
                       AVG(budget) as avg_budget,
                       AVG(revenue) as avg_revenue
                FROM movies 
                WHERE release_year BETWEEN 2010 AND 2023
                  AND budget > 0 AND revenue > 0
                GROUP BY release_year 
                ORDER BY release_year DESC;
            """
        },
        {
            'name': 'Complex Key-Value Aggregation',
            'query': """
                SELECT kv1.key_value as rating_category,
                       kv2.key_value as budget_category,
                       COUNT(*) as movie_count,
                       AVG(m.vote_average) as avg_rating
                FROM movie_keyvalue kv1
                JOIN movie_keyvalue kv2 ON kv1.movie_id = kv2.movie_id
                JOIN movies m ON kv1.movie_id = m.movie_id
                WHERE kv1.key_name = 'rating_category' 
                  AND kv2.key_name = 'budget_category'
                  AND m.budget > 0
                GROUP BY kv1.key_value, kv2.key_value
                ORDER BY movie_count DESC;
            """
        },
        {
            'name': 'Text Search',
            'query': """
                SELECT movie_id, title, vote_average, overview
                FROM movies 
                WHERE title ILIKE '%action%' 
                   OR overview ILIKE '%action%'
                ORDER BY vote_average DESC
                LIMIT 50;
            """
        },
        {
            'name': 'Complex Analysis with Multiple Tables',
            'query': """
                WITH genre_analysis AS (
                    SELECT 
                        TRIM(unnest(string_to_array(m.genres, ','))) as genre,
                        m.movie_id,
                        m.vote_average,
                        kv.key_value as rating_category
                    FROM movies m
                    LEFT JOIN movie_keyvalue kv ON m.movie_id = kv.movie_id 
                        AND kv.key_name = 'rating_category'
                    WHERE m.genres IS NOT NULL 
                      AND m.release_year >= 2010
                )
                SELECT genre,
                       COUNT(*) as total_movies,
                       ROUND(AVG(vote_average), 2) as avg_rating,
                       COUNT(CASE WHEN rating_category = 'excellent' THEN 1 END) as excellent_movies
                FROM genre_analysis
                GROUP BY genre
                HAVING COUNT(*) >= 10
                ORDER BY avg_rating DESC
                LIMIT 15;
            """
        }
    ]

    # Execute each test
    for test in tests:
        execution_time, row_count, query_results = execute_query_with_timing(
            cursor, test['query'], test['name']
        )

        results.append({
            'test_name': test['name'],
            'execution_time': execution_time,
            'row_count': row_count,
            'timestamp': datetime.now().isoformat()
        })

        # Display sample results
        if query_results:
            print("Sample results:")
            for i, row in enumerate(query_results[:3]):
                print(f"  {i + 1}: {row}")

    # Save results to file
    with open('../results/04_performance_test/performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Create performance chart
    create_performance_chart(results)

    cursor.close()
    conn.close()

    return results


def create_performance_chart(results):
    """Create a bar chart of query performance"""
    test_names = [r['test_name'] for r in results]
    execution_times = [r['execution_time'] for r in results]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(test_names)), execution_times, color='skyblue', edgecolor='navy')

    plt.title('PostgreSQL Key-Value Performance Test Results', fontsize=16, fontweight='bold')
    plt.xlabel('Test Cases', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(range(len(test_names)),
               [name[:30] + '...' if len(name) > 30 else name for name in test_names],
               rotation=45, ha='right')

    # Add value labels to each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                 f'{execution_times[i]:.4f}s',
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('postgresql_performance_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nChart saved as 'postgresql_performance_results.png'")


def test_key_value_vs_traditional():
    """Compare Key-Value approach vs Traditional approach"""
    conn = create_connection()
    if not conn:
        return

    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("COMPARISON: Key-Value vs Traditional Approach")
    print("=" * 60)

    # Key-Value approach
    kv_query = """
        SELECT m.title, kv.key_value as rating_category
        FROM movies m
        JOIN movie_keyvalue kv ON m.movie_id = kv.movie_id
        WHERE kv.key_name = 'rating_category'
          AND m.release_year = 2020
        ORDER BY m.vote_average DESC
        LIMIT 100;
    """

    kv_time, kv_rows, _ = execute_query_with_timing(cursor, kv_query, "Key-Value Approach")

    # Traditional approach
    traditional_query = """
        SELECT title,
               CASE 
                   WHEN vote_average >= 8.0 THEN 'excellent'
                   WHEN vote_average >= 7.0 THEN 'good'
                   WHEN vote_average >= 6.0 THEN 'average'
                   WHEN vote_average >= 5.0 THEN 'below_average'
                   ELSE 'poor'
               END as rating_category
        FROM movies
        WHERE release_year = 2020
        ORDER BY vote_average DESC
        LIMIT 100;
    """

    trad_time, trad_rows, _ = execute_query_with_timing(cursor, traditional_query, "Traditional Approach")

    # Compare results
    print(f"\n{'COMPARISON RESULTS'}")
    print("=" * 40)
    print(f"Key-Value: {kv_time:.4f}s ({kv_rows} rows)")
    print(f"Traditional: {trad_time:.4f}s ({trad_rows} rows)")

    if kv_time < trad_time:
        print(f"Key-Value is faster by {((trad_time - kv_time) / trad_time * 100):.1f}%")
    else:
        print(f"Traditional is faster by {((kv_time - trad_time) / kv_time * 100):.1f}%")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    print("PostgreSQL Key-Value Performance Testing")
    print("=" * 50)

    # Run basic performance tests
    results = run_performance_tests()

    # Compare both approaches
    test_key_value_vs_traditional()

    print(f"\nTesting completed!")
    print(f"Results saved in 'performance_results.json'")
    print(f"Chart saved in 'postgresql_performance_results.png'")
