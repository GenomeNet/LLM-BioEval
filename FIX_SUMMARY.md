# Database Path Fix Summary

## Problem
The admin dashboard wasn't showing existing experiments because it was looking at the wrong database.

## Root Cause
- The original config uses `microbellm_jobs.db` as the main database
- When creating `shared.py`, I incorrectly hardcoded the path to `microbellm.db`
- This caused the admin app to look at an empty/different database

## Solution
Fixed `shared.py` to use the correct database path from config:
```python
# Before (incorrect):
DATABASE_PATH = str(PROJECT_ROOT / 'microbellm.db')

# After (correct):
DATABASE_PATH = str(PROJECT_ROOT / config.DATABASE_PATH)  # Uses microbellm_jobs.db
```

## Result
- Admin dashboard now correctly shows all 155 existing combinations
- All 45 models and 2 species files are visible
- The dashboard should work exactly as before

## Database Info
- **Main database**: `microbellm_jobs.db` (51MB)
- Contains: combinations, species_results, managed_models, etc.
- Has 155 existing experimental combinations

The admin dashboard should now work properly and display all your existing experiments!