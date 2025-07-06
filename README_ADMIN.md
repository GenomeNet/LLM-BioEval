# MicrobeLLM Web Applications

The MicrobeLLM project now has two separate web applications:

## 1. Public Website (`microbellm-web`)
The public-facing website for displaying research results and visualizations.

### Features:
- Home page with project overview
- Research pages (Knowledge Calibration, Phenotype Analysis)
- About, Privacy, and Imprint pages
- Read-only data visualizations
- No authentication required

### Running the Public Site:
```bash
# Default: runs on port 5000
microbellm-web

# With custom options
microbellm-web --port 8080 --debug

# Or using the wrapper script
python run_web.py
```

## 2. Admin Dashboard (`microbellm-admin`)
Local administration interface for managing LLM processing jobs.

### Features:
- Dashboard for managing processing combinations
- Import/Export functionality
- Template management
- Settings and API key configuration
- Real-time job monitoring with WebSocket support
- Start/stop/pause job controls

### Running the Admin Dashboard:
```bash
# Default: runs on localhost:5050
microbellm-admin

# With custom options
microbellm-admin --port 5055 --debug

# Or using the wrapper script
python run_admin.py
```

## Important Notes:

1. **API Key Required**: Both applications need the `OPENROUTER_API_KEY` environment variable set for LLM functionality:
   ```bash
   export OPENROUTER_API_KEY='your-api-key-here'
   ```

2. **Database**: Both applications share the same SQLite databases:
   - `microbebench.db` - Main results database
   - `microbellm_jobs.db` - Job tracking database

3. **Security**: The admin dashboard is configured to run on localhost only by default. Do not expose it to the public internet.

4. **Development**: When developing, you can run both applications simultaneously on different ports:
   ```bash
   # Terminal 1: Public site
   microbellm-web --debug
   
   # Terminal 2: Admin dashboard
   microbellm-admin --debug
   ```

## Architecture Benefits:

- **Separation of Concerns**: Public and admin functionality are clearly separated
- **Security**: Admin interface can be kept completely private
- **Deployment**: Deploy only the public site to production servers
- **Performance**: Public site doesn't carry the overhead of job processing code
- **Maintenance**: Easier to maintain and update each component independently