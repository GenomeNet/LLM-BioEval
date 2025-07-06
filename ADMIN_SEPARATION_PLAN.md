# Project Plan: Separate Public and Admin Web Applications

## Objective
Split the monolithic web application into two separate Flask applications:
1. Public site (microbellm-web) - for the production website
2. Admin dashboard (microbellm-admin) - for local administration

## Completed Tasks ✓
- [x] Created shared.py for common utilities
- [x] Created admin_app.py with Flask app and routes for admin functionality
- [x] Created run_admin.py wrapper script
- [x] Updated setup.py to add microbellm-admin entry point
- [x] Extracted admin routes to admin_app.py
- [x] Added ProcessingManager implementation to admin_app.py

## Remaining Tasks
- [ ] Update web_app.py to remove admin functionality
- [ ] Test both applications work independently
- [ ] Consider extracting ProcessingManager to a separate module (future improvement)

## Changes Needed for web_app.py

### Lines to Remove/Comment:
1. **ProcessingManager class** (lines 77-1637) - This entire class should be removed
2. **Global processing_manager variable** and initialization
3. **Admin routes to remove:**
   - `/dashboard` (line 1775)
   - `/api/start_combination/<int:combination_id>` (line 1787)
   - `/api/restart_combination/<int:combination_id>` (line 1794)
   - `/api/pause_combination/<int:combination_id>` (line 1802)
   - `/api/stop_combination/<int:combination_id>` (line 1810)
   - `/api/set_rate_limit` (line 1818)
   - `/api/get_settings` (line 1835)
   - `/api/dashboard_data` (line 1843)
   - `/api/delete_combination/<int:combination_id>` (line 1849)
   - `/export` (line 1881)
   - `/import` (line 1890)
   - `/settings` (line 2064)
   - `/templates` (line 2069)
   - `/api/export_csv` (line 3122)
   - `/api/import_csv` (line 3148)
   - `/api/create_combination` (line 3180)
   - `/api/add_model` (line 3248)
   - `/api/delete_model` (line 3266)
   - `/api/add_species_file` (line 3284)
   - `/api/delete_species_file` (line 3302)
   - `/api/combination_details/<int:combination_id>` (line 3320)
   - `/api/api_key_status` (line 3430)
   - `/api/set_api_key` (line 3462)
   - `/api/get_openrouter_models` (line 3507)
   - `/api/validate_model` (line 3578)
   - `/view_template/<template_type>/<template_name>` (line 3658)
   - `/api/rerun_failed_species` (line 3744)

### Routes to Keep:
- `/` (Home page)
- `/research`
- `/about`
- `/imprint`
- `/privacy`
- `/knowledge_calibration` and `/hallucination_test`
- `/phenotype_analysis`
- `/search_correlation`
- `/correlation`
- `/compare`
- `/components`
- All public data APIs

## How to Run After Changes

### Public Website:
```bash
# Default port 5000
microbellm-web

# Or with custom options
microbellm-web --port 8080 --debug
```

### Admin Dashboard:
```bash
# Default port 5050, localhost only
microbellm-admin

# Or with custom options
microbellm-admin --port 5055 --debug
```

## Review
The separation provides clear benefits:
- Security: Admin interface is never exposed to the web
- Deployment: Can deploy only the public site
- Maintenance: Easier to maintain and develop each part independently
- Performance: Public site doesn't carry the overhead of job processing code