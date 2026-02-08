# Task Generation Module

Scripts for expanding and evolving the task dataset: collecting URLs from evaluation benchmarks, expanding URL coverage via web search, and generating new tasks in parallel.

## Components

- **`collect_url.py`**: Collect unique URLs from evaluation datasets
  - Extract URLs from various benchmark configs
  - Categorize websites by type
  - Filter and deduplicate URLs

- **`expand_urls.py`**: Expand URL coverage using web search
  - Query-based URL discovery via SerpAPI
  - Category-based URL expansion
  - Search result aggregation

- **`generate_tasks_parallel.py`**: Generate new tasks in parallel

## Features

### URL Collection
- Aggregate URLs from multiple evaluation benchmarks
- Maintain URL-to-query mappings
- Filter by website categories

### Task Expansion
- Use search engines to find related websites
- Generate diverse task descriptions
- Category-based task generation

## Usage

### Collect URLs

```python
# Run collect_url.py to extract URLs from configs
# Output: website_set dictionary with URLs and queries
```

### Expand with Search

**Note:** This requires a SerpAPI key. Set in environment or `.env`:
```bash
SERPAPI_API_KEY=your_api_key_here
```

```python
# Run expand_urls.py to expand URL coverage
# Configurable parameters:
# - MAX_QUERIES_PER_SITE: Number of queries per site
# - SEARCH_RESULTS_PER_QUERY: Results per query
# - MAX_SERPAPI_QUERIES: Total query budget
```

## Output

Generated data is written under the `task_generation/` directory (e.g. `generated_tasks/`, logs). It includes:
- Expanded website categories
- New task descriptions
- URL-to-category mappings
- Task configuration files

## Categories

Supported website categories:
- shopping, travel, academic, news
- government, health, entertainment
- finance, food, tech, education
- career, social, media, sports, services

## Notes

- Set `INFERENCE_PROJECT_ROOT` if running scripts from a different working directory
- SerpAPI key required for URL expansion (`SERPAPI_API_KEY`)
- Rate limiting applies for search APIs
