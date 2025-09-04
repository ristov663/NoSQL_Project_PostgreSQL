import psycopg2
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

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


def execute_query_with_timing(cursor, query, description, fetch_results=True):
    """Execute a query and measure execution time"""
    print(f"\n📊 Executing: {description}")
    print("=" * 60)

    start_time = time.time()
    cursor.execute(query)

    if fetch_results:
        results = cursor.fetchall()
        row_count = len(results)
    else:
        results = None
        row_count = cursor.rowcount

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"⏱️  Execution time: {execution_time:.4f} seconds")
    print(f"📈 Number of results: {row_count}")

    if results and len(results) > 0:
        print("🔍 Sample results:")
        for i, row in enumerate(results[:3]):
            print(f"   {i + 1}: {row}")

    return execution_time, row_count, results


def run_comprehensive_performance_tests():
    """Run comprehensive performance tests"""
    conn = create_connection()
    if not conn:
        return

    cursor = conn.cursor()
    results = []

    print("🚀 STARTING COMPREHENSIVE PERFORMANCE TESTING")
    print("=" * 80)

    # Define all test queries
    tests = [
        # ============== SIMPLE QUERIES ==============
        {
            'category': 'Simple',
            'name': 'Top Rated Movies',
            'query': """
                SELECT movie_id, title, vote_average, vote_count, release_year
                FROM movies 
                WHERE vote_average > 8.0 
                ORDER BY vote_average DESC, vote_count DESC 
                LIMIT 50;
            """
        },
        {
            'category': 'Simple',
            'name': 'Movies from 2020',
            'query': """
                SELECT movie_id, title, vote_average, budget, revenue
                FROM movies 
                WHERE release_year = 2020 
                ORDER BY vote_average DESC 
                LIMIT 30;
            """
        },
        {
            'category': 'Simple',
            'name': 'High Budget Movies',
            'query': """
                SELECT movie_id, title, budget, revenue, (revenue - budget) as profit
                FROM movies 
                WHERE budget > 100000000 
                ORDER BY budget DESC 
                LIMIT 25;
            """
        },
        {
            'category': 'Simple',
            'name': 'Long Movies',
            'query': """
                SELECT movie_id, title, runtime, vote_average, release_year
                FROM movies 
                WHERE runtime > 180
                ORDER BY runtime DESC
                LIMIT 20;
            """
        },
        {
            'category': 'Simple',
            'name': 'Recent Movies',
            'query': """
                SELECT movie_id, title, release_date, vote_average, popularity
                FROM movies 
                WHERE release_year >= 2020
                ORDER BY release_date DESC
                LIMIT 40;
            """
        },

        # ============== COMPLEX QUERIES ==============
        {
            'category': 'Complex',
            'name': 'Key-Value Join - Basic',
            'query': """
                SELECT m.movie_id, m.title, m.vote_average, m.budget,
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
            'category': 'Complex',
            'name': 'Runtime Category Analysis',
            'query': """
                SELECT m.movie_id, m.title, m.runtime, m.budget, m.revenue,
                       kv_runtime.key_value as runtime_category,
                       CASE 
                           WHEN m.revenue > m.budget * 2 THEN 'highly_profitable'
                           WHEN m.revenue > m.budget THEN 'profitable'
                           ELSE 'not_profitable'
                       END as profitability
                FROM movies m
                LEFT JOIN movie_keyvalue kv_runtime ON m.movie_id = kv_runtime.movie_id 
                    AND kv_runtime.key_name = 'runtime_category'
                WHERE m.budget > 0 AND m.revenue > 0 AND m.release_year >= 2010
                ORDER BY (m.revenue - m.budget) DESC
                LIMIT 50;
            """
        },
        {
            'category': 'Complex',
            'name': 'Multi Key-Value Analysis',
            'query': """
        SELECT kv1.key_value as rating_category,
               kv2.key_value as budget_category,
               kv3.key_value as runtime_category,
               COUNT(*) as movie_count,
               ROUND(AVG(m.vote_average)::numeric, 2) as avg_rating,
               ROUND(AVG(m.budget)::numeric, 2) as avg_budget
        FROM movie_keyvalue kv1
        JOIN movie_keyvalue kv2 ON kv1.movie_id = kv2.movie_id
        JOIN movie_keyvalue kv3 ON kv1.movie_id = kv3.movie_id
        JOIN movies m ON kv1.movie_id = m.movie_id
        WHERE kv1.key_name = 'rating_category' 
          AND kv2.key_name = 'budget_category'
          AND kv3.key_name = 'runtime_category'
          AND m.budget > 0
        GROUP BY kv1.key_value, kv2.key_value, kv3.key_value
        HAVING COUNT(*) >= 5
        ORDER BY movie_count DESC;
    """

        },
        {
            'category': 'Complex',
            'name': 'Text Search',
            'query': """
                SELECT movie_id, title, vote_average, overview
                FROM movies 
                WHERE (title ILIKE '%action%' OR overview ILIKE '%action%'
                    OR title ILIKE '%adventure%' OR overview ILIKE '%adventure%')
                  AND vote_average > 6.0
                ORDER BY vote_average DESC
                LIMIT 75;
            """
        },
        {
            'category': 'Complex',
            'name': 'Financial Analysis',
            'query': """
    SELECT movie_id, title, budget, revenue,
           (revenue - budget) as profit,
           ROUND((revenue::numeric / NULLIF(budget::numeric, 0)), 2) as roi,
           release_year,
           CASE 
               WHEN revenue > budget * 3 THEN 'blockbuster'
               WHEN revenue > budget * 2 THEN 'very_successful'
               WHEN revenue > budget THEN 'successful'
               ELSE 'unsuccessful'
           END as success_category
    FROM movies
    WHERE budget > 1000000 AND revenue > 0
    ORDER BY (revenue - budget) DESC
    LIMIT 60;
"""

        },

        # ============== VERY COMPLEX QUERIES ==============
        {
            'category': 'Very Complex',
            'name': 'Detailed Yearly Report',
            'query': """
    SELECT release_year,
           COUNT(*) as total_movies,
           ROUND(AVG(vote_average)::numeric, 2) as avg_rating,
           ROUND(AVG(budget)::numeric, 2) as avg_budget,
           ROUND(AVG(revenue)::numeric, 2) as avg_revenue,
           ROUND(SUM(revenue)::numeric, 2) as total_revenue,
           ROUND(AVG(runtime)::numeric, 2) as avg_runtime,
           COUNT(CASE WHEN vote_average > 7.0 THEN 1 END) as high_rated_movies,
           COUNT(CASE WHEN vote_average > 8.0 THEN 1 END) as excellent_movies,
           ROUND(AVG((revenue - budget))::numeric, 2) as avg_profit,
           ROUND(AVG(popularity)::numeric, 2) as avg_popularity
    FROM movies 
    WHERE release_year IS NOT NULL 
      AND release_year BETWEEN 2000 AND 2023
      AND budget > 0 AND revenue > 0
    GROUP BY release_year 
    ORDER BY release_year DESC;
"""

        },
        {
            'category': 'Very Complex',
            'name': 'Production Companies Analysis',
            'query': """
    WITH company_stats AS (
        SELECT 
            TRIM(unnest(string_to_array(production_companies, ','))) as company,
            COUNT(*) as movie_count,
            AVG(vote_average)::numeric as avg_rating,
            SUM(revenue)::numeric as total_revenue,
            AVG(budget)::numeric as avg_budget,
            SUM(revenue - budget)::numeric as total_profit,
            AVG(popularity)::numeric as avg_popularity
        FROM movies 
        WHERE production_companies IS NOT NULL 
          AND production_companies != ''
          AND budget > 0 AND revenue > 0
          AND release_year >= 2010
        GROUP BY company
    )
    SELECT company, movie_count, 
           ROUND(avg_rating, 2) as avg_rating,
           ROUND(total_revenue / 1000000, 2) as total_revenue_millions,
           ROUND(avg_budget / 1000000, 2) as avg_budget_millions,
           ROUND(total_profit / 1000000, 2) as total_profit_millions,
           ROUND(total_profit / movie_count / 1000000, 2) as avg_profit_per_movie_millions,
           ROUND(avg_popularity, 2) as avg_popularity
    FROM company_stats 
    WHERE movie_count >= 10
    ORDER BY total_revenue DESC 
    LIMIT 20;
"""

        },
        {
            'category': 'Very Complex',
            'name': 'Genre Analysis with Key-Value',
            'query': """
    WITH genre_analysis AS (
        SELECT 
            TRIM(unnest(string_to_array(m.genres, ','))) as genre,
            m.movie_id,
            m.vote_average,
            m.budget,
            m.revenue,
            m.runtime,
            m.popularity,
            kv.key_value as rating_category
        FROM movies m
        LEFT JOIN movie_keyvalue kv ON m.movie_id = kv.movie_id 
            AND kv.key_name = 'rating_category'
        WHERE m.genres IS NOT NULL 
          AND m.genres != ''
          AND m.release_year >= 2000
    )
    SELECT genre,
           COUNT(*) as total_movies,
           ROUND(AVG(vote_average)::numeric, 2) as avg_rating,
           COUNT(CASE WHEN rating_category = 'excellent' THEN 1 END) as excellent_movies,
           COUNT(CASE WHEN rating_category = 'good' THEN 1 END) as good_movies,
           COUNT(CASE WHEN rating_category = 'average' THEN 1 END) as average_movies,
           ROUND((AVG(budget)::numeric) / 1000000, 2) as avg_budget_millions,
           ROUND((AVG(revenue)::numeric) / 1000000, 2) as avg_revenue_millions,
           ROUND(AVG(runtime)::numeric, 2) as avg_runtime,
           ROUND(AVG(popularity)::numeric, 2) as avg_popularity,
           ROUND((SUM(revenue - budget)::numeric) / 1000000, 2) as total_profit_millions
    FROM genre_analysis
    WHERE budget > 0 AND revenue > 0
    GROUP BY genre
    HAVING COUNT(*) >= 20
    ORDER BY avg_rating DESC, total_movies DESC
    LIMIT 25;
"""

        },
        {
            'category': 'Very Complex',
            'name': 'Complex Time Series',
            'query': """
    SELECT 
        m.release_year,
        COUNT(*) as total_movies,

        -- Rating analysis
        ROUND(AVG(m.vote_average)::numeric, 2) as avg_rating,
        COUNT(CASE WHEN kv_rating.key_value = 'excellent' THEN 1 END) as excellent_count,
        COUNT(CASE WHEN kv_rating.key_value = 'good' THEN 1 END) as good_count,
        COUNT(CASE WHEN kv_rating.key_value = 'average' THEN 1 END) as average_count,

        -- Budget analysis  
        ROUND((AVG(m.budget)::numeric) / 1000000, 2) as avg_budget_millions,
        COUNT(CASE WHEN kv_budget.key_value = 'blockbuster' THEN 1 END) as blockbuster_count,
        COUNT(CASE WHEN kv_budget.key_value = 'big_budget' THEN 1 END) as big_budget_count,
        COUNT(CASE WHEN kv_budget.key_value = 'medium_budget' THEN 1 END) as medium_budget_count,

        -- Runtime analysis
        ROUND(AVG(m.runtime)::numeric, 2) as avg_runtime,
        COUNT(CASE WHEN kv_runtime.key_value = 'very_long' THEN 1 END) as very_long_movies,
        COUNT(CASE WHEN kv_runtime.key_value = 'long' THEN 1 END) as long_movies,
        COUNT(CASE WHEN kv_runtime.key_value = 'standard' THEN 1 END) as standard_movies,

        -- Financial analysis
        ROUND((SUM(m.revenue - m.budget)::numeric) / 1000000, 2) as total_profit_millions,
        ROUND(AVG(CASE WHEN m.revenue > 0 THEN (m.revenue::numeric / NULLIF(m.budget::numeric, 0)) END)::numeric, 2) as avg_roi,
        ROUND(AVG(m.popularity)::numeric, 2) as avg_popularity

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
"""

        },
        {
            'category': 'Very Complex',
            'name': 'Actors and Crew Analysis',
            'query': """
    WITH cast_analysis AS (
        SELECT 
            TRIM(unnest(string_to_array(star1 || ',' || star2 || ',' || star3 || ',' || star4, ','))) as actor,
            movie_id,
            vote_average,
            budget,
            revenue
        FROM movies
        WHERE (star1 IS NOT NULL OR star2 IS NOT NULL OR star3 IS NOT NULL OR star4 IS NOT NULL)
          AND budget > 0 AND revenue > 0
          AND release_year >= 2010
    )
    SELECT actor,
           COUNT(*) as movie_count,
           ROUND(AVG(vote_average)::numeric, 2) as avg_rating,
           ROUND((AVG(budget)::numeric) / 1000000, 2) as avg_budget_millions,
           ROUND((AVG(revenue)::numeric) / 1000000, 2) as avg_revenue_millions,
           ROUND((SUM(revenue - budget)::numeric) / 1000000, 2) as total_profit_millions
    FROM cast_analysis
    WHERE actor != '' AND actor IS NOT NULL
    GROUP BY actor
    HAVING COUNT(*) >= 5
    ORDER BY total_profit_millions DESC
    LIMIT 30;
"""

        },

        # ============== KEY-VALUE SPECIFIC TESTS ==============
        {
            'category': 'Key-Value',
            'name': 'Key-Value Aggregation by Category',
            'query': """
                SELECT key_name, key_value, 
       COUNT(*) as count,
       ROUND(AVG(m.vote_average)::numeric, 2) as avg_rating,
       ROUND((AVG(m.budget)::numeric) / 1000000, 2) as avg_budget_millions,
       ROUND((AVG(m.revenue)::numeric) / 1000000, 2) as avg_revenue_millions
FROM movie_keyvalue kv
JOIN movies m ON kv.movie_id = m.movie_id
WHERE m.budget > 0 AND m.revenue > 0
GROUP BY key_name, key_value
ORDER BY key_name, count DESC;
            """
        },
        {
            'category': 'Key-Value',
            'name': 'Key-Value Time Series',
            'query': """
                SELECT m.release_year, kv.key_name, kv.key_value,
       COUNT(*) as count,
       ROUND(AVG(m.vote_average)::numeric, 2) as avg_rating
FROM movie_keyvalue kv
JOIN movies m ON kv.movie_id = m.movie_id
WHERE m.release_year BETWEEN 2015 AND 2023
GROUP BY m.release_year, kv.key_name, kv.key_value
HAVING COUNT(*) >= 10
ORDER BY m.release_year, kv.key_name, count DESC;
            """
        }
    ]

    # Execute all tests
    for test in tests:
        execution_time, row_count, query_results = execute_query_with_timing(
            cursor, test['query'], f"{test['category']}: {test['name']}"
        )

        results.append({
            'category': test['category'],
            'test_name': test['name'],
            'execution_time': execution_time,
            'row_count': row_count,
            'timestamp': datetime.now().isoformat()
        })

    # Save results
    with open('../results/05_performance_test/comprehensive_performance_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    cursor.close()
    conn.close()

    return results


def create_comprehensive_visualizations(results):
    """Create comprehensive visualizations"""

    # Prepare data
    df = pd.DataFrame(results)

    # Set style
    plt.style.use('seaborn-v0_8')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

    print("\n📊 CREATING VISUALIZATIONS...")

    # ============== Visualization 1: Overview of all performance tests ==============
    fig, ax = plt.subplots(figsize=(16, 10))

    test_names = [r['test_name'][:30] + '...' if len(r['test_name']) > 30 else r['test_name'] for r in results]
    execution_times = [r['execution_time'] for r in results]
    categories = [r['category'] for r in results]

    # Create color map per category
    unique_categories = list(set(categories))
    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}
    bar_colors = [color_map[cat] for cat in categories]

    bars = ax.bar(range(len(test_names)), execution_times, color=bar_colors, alpha=0.8, edgecolor='black')

    ax.set_title('PostgreSQL Key-Value Performance Test Results\nAll Query Categories',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Queries', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(test_names)))
    ax.set_xticklabels(test_names, rotation=45, ha='right', fontsize=10)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(execution_times) * 0.01,
                f'{execution_times[i]:.3f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Legend for categories
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[cat], alpha=0.8, label=cat)
                       for cat in unique_categories]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('01_all_performance_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ============== Visualization 2: Category Comparison ==============
    fig, ax = plt.subplots(figsize=(14, 8))

    category_stats = df.groupby('category').agg({
        'execution_time': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)

    categories = category_stats.index
    means = category_stats['execution_time']['mean']
    stds = category_stats['execution_time']['std']

    bars = ax.bar(categories, means, yerr=stds, capsize=5, color=colors[:len(categories)],
                  alpha=0.8, edgecolor='black')

    ax.set_title('Average Execution Time by Query Category\nwith Standard Deviation',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Categories', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Time (seconds)', fontsize=14, fontweight='bold')

    # Add values on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + stds.iloc[i] + max(means) * 0.02,
                f'{means.iloc[i]:.3f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Add test counts
        ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                f'{int(category_stats["execution_time"]["count"].iloc[i])} tests',
                ha='center', va='center', fontweight='bold', fontsize=10, color='white')

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('02_category_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ============== Visualization 3: Box Plot by Category ==============
    fig, ax = plt.subplots(figsize=(12, 8))

    categories_data = [df[df['category'] == cat]['execution_time'].tolist() for cat in unique_categories]

    box_plot = ax.boxplot(categories_data, labels=unique_categories, patch_artist=True)

    # Color the boxplots
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_title('Execution Time Distribution by Category',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Categories', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('03_execution_time_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ============== Visualization 4: Scatter Plot - Time vs Result Count ==============
    fig, ax = plt.subplots(figsize=(14, 10))

    for i, category in enumerate(unique_categories):
        cat_data = df[df['category'] == category]
        ax.scatter(cat_data['row_count'], cat_data['execution_time'],
                   c=colors[i], label=category, alpha=0.7, s=100, edgecolors='black')

    ax.set_title('Correlation Between Result Count and Execution Time',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Results', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add trend line
    x = df['row_count']
    y = df['execution_time']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label='Trend Line')

    plt.tight_layout()
    plt.savefig('04_time_vs_results_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ============== Visualization 5: Performance Heatmap ==============
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create pivot table
    pivot_data = df.pivot_table(values='execution_time',
                                index='category',
                                columns=df.groupby('category').cumcount(),
                                fill_value=0)

    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Execution Time (seconds)'}, ax=ax)

    ax.set_title('Heatmap of Execution Times by Category and Test Number',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Test Number within Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Categories', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('05_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ============== Visualization 6: Detailed Pie Charts ==============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Pie chart for category distribution
    category_counts = df['category'].value_counts()
    ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
            colors=colors[:len(category_counts)], startangle=90)
    ax1.set_title('Test Distribution by Category', fontsize=14, fontweight='bold')

    # Pie chart for execution time distribution
    category_times = df.groupby('category')['execution_time'].sum()
    ax2.pie(category_times.values, labels=category_times.index, autopct='%1.1f%%',
            colors=colors[:len(category_times)], startangle=90)
    ax2.set_title('Total Execution Time Distribution by Category', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('06_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ All visualizations created and saved!")

    return df


def detailed_performance_analysis(results):
    """Detailed performance analysis"""
    df = pd.DataFrame(results)

    print("\n📈 DETAILED PERFORMANCE ANALYSIS")
    print("=" * 60)

    # General statistics
    print(f"🔢 Total tests: {len(results)}")
    print(f"⏱️  Total execution time: {df['execution_time'].sum():.4f} seconds")
    print(f"📊 Average execution time: {df['execution_time'].mean():.4f} seconds")
    print(f"📈 Median execution time: {df['execution_time'].median():.4f} seconds")
    print(f"🎯 Fastest test: {df.loc[df['execution_time'].idxmin(), 'test_name']} ({df['execution_time'].min():.4f}s)")
    print(f"🐌 Slowest test: {df.loc[df['execution_time'].idxmax(), 'test_name']} ({df['execution_time'].max():.4f}s)")

    # Analysis by category
    print(f"\n📋 ANALYSIS BY CATEGORY:")
    category_analysis = df.groupby('category').agg({
        'execution_time': ['count', 'mean', 'std', 'min', 'max', 'sum'],
        'row_count': ['mean', 'sum']
    }).round(4)

    for category in category_analysis.index:
        stats = category_analysis.loc[category]
        print(f"\n  🏷️  {category}:")
        print(f"    • Number of tests: {int(stats['execution_time']['count'])}")
        print(f"    • Average time: {stats['execution_time']['mean']:.4f}s")
        print(f"    • Standard deviation: {stats['execution_time']['std']:.4f}s")
        print(f"    • Fastest: {stats['execution_time']['min']:.4f}s")
        print(f"    • Slowest: {stats['execution_time']['max']:.4f}s")
        print(f"    • Total time: {stats['execution_time']['sum']:.4f}s")
        print(f"    • Average result count: {stats['row_count']['mean']:.0f}")

    return df


def test_key_value_vs_traditional():
    """Detailed comparison of Key-Value vs Traditional approach"""
    conn = create_connection()
    if not conn:
        return

    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("🥊 DETAILED COMPARISON: Key-Value vs Traditional Approach")
    print("=" * 80)

    comparison_tests = [
        {
            'name': 'Simple Categorization',
            'kv_query': """
                SELECT m.title, kv.key_value as rating_category
                FROM movies m
                JOIN movie_keyvalue kv ON m.movie_id = kv.movie_id
                WHERE kv.key_name = 'rating_category'
                  AND m.release_year = 2020
                ORDER BY m.vote_average DESC
                LIMIT 100;
            """,
            'traditional_query': """
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
        },
        {
            'name': 'Complex Categorization',
            'kv_query': """
                SELECT m.title, 
                       kv1.key_value as rating_category,
                       kv2.key_value as budget_category
                FROM movies m
                JOIN movie_keyvalue kv1 ON m.movie_id = kv1.movie_id AND kv1.key_name = 'rating_category'
                JOIN movie_keyvalue kv2 ON m.movie_id = kv2.movie_id AND kv2.key_name = 'budget_category'
                WHERE m.release_year >= 2018
                ORDER BY m.vote_average DESC
                LIMIT 200;
            """,
            'traditional_query': """
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
                WHERE release_year >= 2018
                ORDER BY vote_average DESC
                LIMIT 200;
            """
        },
        {
            'name': 'Aggregation by Categories',
            'kv_query': """
                SELECT kv.key_value as category, COUNT(*) as count, AVG(m.vote_average) as avg_rating
                FROM movie_keyvalue kv
                JOIN movies m ON kv.movie_id = m.movie_id
                WHERE kv.key_name = 'rating_category'
                  AND m.release_year >= 2015
                GROUP BY kv.key_value
                ORDER BY count DESC;
            """,
            'traditional_query': """
                SELECT 
                    CASE 
                        WHEN vote_average >= 8.0 THEN 'excellent'
                        WHEN vote_average >= 7.0 THEN 'good'
                        WHEN vote_average >= 6.0 THEN 'average'
                        WHEN vote_average >= 5.0 THEN 'below_average'
                        ELSE 'poor'
                    END as category,
                    COUNT(*) as count,
                    AVG(vote_average) as avg_rating
                FROM movies
                WHERE release_year >= 2015
                GROUP BY 
                    CASE 
                        WHEN vote_average >= 8.0 THEN 'excellent'
                        WHEN vote_average >= 7.0 THEN 'good'
                        WHEN vote_average >= 6.0 THEN 'average'
                        WHEN vote_average >= 5.0 THEN 'below_average'
                        ELSE 'poor'
                    END
                ORDER BY count DESC;
            """
        }
    ]

    comparison_results = []

    for test in comparison_tests:
        print(f"\n🔍 Test: {test['name']}")
        print("-" * 50)

        # Key-Value approach
        kv_time, kv_rows, _ = execute_query_with_timing(cursor, test['kv_query'],
                                                        f"Key-Value - {test['name']}", fetch_results=False)

        # Traditional approach
        trad_time, trad_rows, _ = execute_query_with_timing(cursor, test['traditional_query'],
                                                            f"Traditional - {test['name']}", fetch_results=False)

        # Initialize winner and improvement with None or default values
        winner = None
        improvement = None

        # Compare results only if trad_time is positive
        if trad_time > 0:
            if kv_time < trad_time:
                winner = "Key-Value"
                improvement = ((trad_time - kv_time) / trad_time * 100)
                print(f"🏆 Winner: {winner} (faster by {improvement:.1f}%)")
            else:
                winner = "Traditional"
                improvement = ((kv_time - trad_time) / kv_time * 100)
                print(f"🏆 Winner: {winner} (faster by {improvement:.1f}%)")
        else:
            print("⚠️ Traditional query execution time is zero or invalid; skipping comparison.")

        comparison_results.append({
            'test_name': test['name'],
            'kv_time': kv_time,
            'traditional_time': trad_time,
            'kv_rows': kv_rows,
            'traditional_rows': trad_rows,
            'winner': winner,
            'improvement': improvement
        })

    # Create visualization for comparison
    create_comparison_visualization(comparison_results)

    cursor.close()
    conn.close()

    return comparison_results


def create_comparison_visualization(comparison_results):
    """Create a visualization for the comparison of approaches"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Prepare data
    test_names = [r['test_name'] for r in comparison_results]
    kv_times = [r['kv_time'] for r in comparison_results]
    trad_times = [r['traditional_time'] for r in comparison_results]

    x = np.arange(len(test_names))
    width = 0.35

    # Bar chart comparison
    bars1 = ax1.bar(x - width / 2, kv_times, width, label='Key-Value', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, trad_times, width, label='Traditional', color='#4ECDC4', alpha=0.8)

    ax1.set_title('Execution Time Comparison\nKey-Value vs Traditional Approach',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Tests', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in test_names])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add bar values
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + max(kv_times + trad_times) * 0.01,
                 f'{height:.3f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + max(kv_times + trad_times) * 0.01,
                 f'{height:.3f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Pie chart for winners
    winners = [r['winner'] for r in comparison_results]
    winner_counts = pd.Series(winners).value_counts()

    ax2.pie(winner_counts.values, labels=winner_counts.index, autopct='%1.1f%%',
            colors=['#FF6B6B' if 'Key-Value' in label else '#4ECDC4' for label in winner_counts.index],
            startangle=90)
    ax2.set_title('Winners in Comparison', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('07_kv_vs_traditional_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_performance_summary_report(results, comparison_results):
    """Create a summary performance report"""

    print("\n" + "=" * 80)
    print("📋 PERFORMANCE SUMMARY REPORT")
    print("=" * 80)

    df = pd.DataFrame(results)

    # General metrics
    total_tests = len(results)
    total_time = df['execution_time'].sum()
    avg_time = df['execution_time'].mean()

    print(f"🔢 Total tests executed: {total_tests}")
    print(f"⏱️  Total time for all tests: {total_time:.4f} seconds")
    print(f"📊 Average time per test: {avg_time:.4f} seconds")

    # Best and worst performance
    fastest = df.loc[df['execution_time'].idxmin()]
    slowest = df.loc[df['execution_time'].idxmax()]

    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"   Test: {fastest['test_name']}")
    print(f"   Category: {fastest['category']}")
    print(f"   Time: {fastest['execution_time']:.4f} seconds")
    print(f"   Results: {fastest['row_count']}")

    print(f"\n🐌 WORST PERFORMANCE:")
    print(f"   Test: {slowest['test_name']}")
    print(f"   Category: {slowest['category']}")
    print(f"   Time: {slowest['execution_time']:.4f} seconds")
    print(f"   Results: {slowest['row_count']}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    # Analyze by category
    category_stats = df.groupby('category')['execution_time'].agg(['mean', 'std']).round(4)
    fastest_category = category_stats['mean'].idxmin()
    slowest_category = category_stats['mean'].idxmax()

    print(f"   ✅ Fastest category: {fastest_category} (avg: {category_stats.loc[fastest_category, 'mean']:.4f}s)")
    print(
        f"   ⚠️  Slowest category: {slowest_category} (avg: {category_stats.loc[slowest_category, 'mean']:.4f}s)")

    if comparison_results:
        kv_wins = sum(1 for r in comparison_results if r['winner'] == 'Key-Value')
        trad_wins = len(comparison_results) - kv_wins

        print(f"\n🥊 APPROACH COMPARISON:")
        print(f"   Key-Value wins: {kv_wins}/{len(comparison_results)}")
        print(f"   Traditional wins: {trad_wins}/{len(comparison_results)}")

        if kv_wins > trad_wins:
            print(f"   🏆 Overall faster: Key-Value approach")
        elif trad_wins > kv_wins:
            print(f"   🏆 Overall faster: Traditional approach")
        else:
            print(f"   🤝 Both approaches performed similarly")

    print(f"\n📊 CONCLUSIONS:")
    print(f"   • PostgreSQL demonstrated solid performance for Key-Value operations")
    print(f"   • Simple queries execute the fastest")
    print(f"   • Complex aggregations take longer but provide rich insights")
    print(f"   • Key-Value approach offers flexibility for dynamic data")

    # Save detailed report
    report = {
        'summary': {
            'total_tests': total_tests,
            'total_time': total_time,
            'average_time': avg_time,
            'fastest_test': fastest.to_dict(),
            'slowest_test': slowest.to_dict()
        },
        'category_analysis': category_stats.to_dict(),
        'comparison_results': comparison_results,
        'timestamp': datetime.now().isoformat()
    }

    with open('../results/05_performance_test/performance_summary_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n💾 Detailed report saved as 'performance_summary_report.json'")


if __name__ == "__main__":
    print("🚀 PostgreSQL Key-Value Comprehensive Performance Testing")
    print("=" * 80)

    try:
        # Run core comprehensive tests
        print("1️⃣  Running core tests...")
        results = run_comprehensive_performance_tests()

        # Create comprehensive visualizations
        print("\n2️⃣  Creating visualizations...")
        df = create_comprehensive_visualizations(results)

        # Detailed analysis
        print("\n3️⃣  Running detailed analysis...")
        detailed_performance_analysis(results)

        # Approach comparison
        print("\n4️⃣  Comparing approaches...")
        comparison_results = test_key_value_vs_traditional()

        # Generate summary report
        print("\n5️⃣  Generating summary report...")
        create_performance_summary_report(results, comparison_results)

        print(f"\n✅ TESTING COMPLETED SUCCESSFULLY!")
        print(f"📂 Saved files:")
        print(f"   • comprehensive_performance_results.json - All Results")
        print(f"   • performance_summary_report.json - Summary Report")
        print(f"   • 01_all_performance_results.png - All Performance Overview")
        print(f"   • 02_category_comparison.png - Category Comparison")
        print(f"   • 03_execution_time_distribution.png - Execution Time Distribution")
        print(f"   • 04_time_vs_results_correlation.png - Time vs Results Correlation")
        print(f"   • 05_performance_heatmap.png - Performance Heatmap")
        print(f"   • 06_distribution_analysis.png - Distribution Analysis")
        print(f"   • 07_kv_vs_traditional_comparison.png - Key-Value vs Traditional")

    except Exception as e:
        print(f"❌ Testing error: {e}")
        import traceback

        traceback.print_exc()
