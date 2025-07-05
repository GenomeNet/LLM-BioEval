# Project plan 

## Project overview 

MicrobeLLM is a Python tool designed to evaluate Large Language Models (LLMs) on their ability to predict microbial phenotypes. This tool helps researchers assess how well different LLMs perform on microbiological tasks by querying them with bacterial species names and comparing their predictions against known characteristics.

## Style guide
- All pages should use a consistent, modern layout as seen in `index.html` and `research.html`.
- The default background for all main content sections is white (`background: var(--bg-primary)`), providing a clean and minimal look.
- Section headers use a flex layout, are left-aligned, and employ small uppercase text with subtle gray coloring (`color: var(--gray-600)`), as in the `.section-header` class.
- Publication and project cards use a white background, rounded corners (`border-radius: 16px`), and a subtle border (`border: 1px solid var(--gray-200)`). Cards have a hover effect with a slight lift and shadow.
- Callout boxes (e.g., hero sections, experimental flow sections) use a gradient background and may span the full width of the viewport. They include top and bottom borders for emphasis and use a blurred background effect for visual separation.
- Only callout boxes should have colored or gradient backgrounds; all other sections remain white.
- Indentation is used only for callout boxes, not for regular content or tables.
- Tables and data sections use white backgrounds, with gray backgrounds reserved for table headers and hover states.
- Spacing between sections and components is managed with CSS variables (e.g., `--spacing-section`, `--spacing-component`) for consistent vertical rhythm.
- All pages are fully responsive, with grid and flex layouts adapting to smaller screens (see media queries in `index.html` and `research.html`).
- Animations (e.g., canvas-based bacteria, DNA, or cell growth) are used for visual interest in cards but do not interfere with content readability.
- Buttons and links use consistent styling, with primary actions in color and secondary actions in gray.
- Footer uses a light gray border and background, with links styled in subtle gray and darkening on hover.

## High-level architecture
- **Frontend**: Flask web application, serving HTML templates and static assets. Main entry point is `microbellm/web_app.py`.
- **Database**: Uses SQLite (`microbebench.db`) to store results, metadata, and user data. The database is created and managed automatically on first run.
- **Environment & Installation**: Setup is managed via `install.sh`, which:
    - Checks for conda and creates a `microbellm` conda environment using `environment.yml`.
    - Installs the package in editable mode (`pip install -e .`).
    - Provides instructions for activating the environment and running the CLI/web interface.
- **Python Packaging**: Defined in `setup.py`:
    - Uses `setuptools` for packaging.
    - Declares dependencies (Flask, pandas, numpy, etc.).
    - Provides console scripts:
        - `microbellm` for CLI tasks (main entry: `microbellm/microbellm.py`)
        - `microbellm-web` for launching the web interface (main entry: `microbellm/web_app.py`)
    - Includes all necessary Python and text/template files as package data.
- **API Key Management**: Requires the `OPENROUTER_API_KEY` environment variable to be set for prediction features. This is checked at runtime (see `server.log` warnings).
- **Server Startup**: The web interface is started with `microbellm-web` (or `microbellm-web --debug --port 5050` for development). The port should be 5050, but can be changed if in use. Access is via `http://localhost:5050`.
- **Logging**: Startup and error messages (such as missing API key or port conflicts) are logged to `server.log`.


## Development checkpoints

### Checkpoint 1: /artificial_dataset page 
- [ ] correct file paths, "Testing LLM knowledge on real and artificial bacterial species" box on `index.html` links to /artificial_dataset but this will display code from `microbellm/templates/knowledge_calibration.html`. So we should rename the url path /artificial_dataset to /knowledge_calibration and update `/research` and `/index` to point to the right paths
- [ ] consistent headlines, it should have the structure, h1 on "Hallucination Check: Fictional Strain Names" and "Web-Aligned Knowledge: Real Bacterial Names vs. Google Counts", this should be maybe visible with a grey line and background gradient (to white) like we had defined before. H2 should be "Top Performing Models" and "Full Results of the Artificial Hallucination Test" (maybe rename it to be more consistent with content). Again, H2 should be "Top Models – Web-Alignment" and "Knowledge-Web Alignment Table" 
- [ ] all full tables should by default not be shown in total, this is currently done for tables under "Full Results of the Artificial Hallucination Test" but not for "Knowledge-Web Alignment Table" 
- [ ] "Top Performing Models" boxes should have the same style, they are currently different for the 2 types 
- [ ] the only indentation should be for "callout boxes" and not for parts of the content. There is indentation that should be removed for "Top Performing Models" 
- [ ] consistent indentation according to style guide, "template" example is not indented as other boxes, "Top Performing Models" is wrongfully indented since it's normal content and not a call-out 
- [ ] add a bar plot of the model QUALITY SCORE, x should be the score and y shoudl be different models. This shoudl be displayed under "Full Results of the Artificial Hallucination Test
".  One bar per template showing all within boxes side by side with annotation we need to understand these. this should show the distribution of models
- [ ] add a similar bar plot for "Knowledge-Web Alignment Table", this time we should show CORRELATION scroes. 
- [ ] since we show here correlation of google search results vs the konwlege group membership we can add a new plot showing the correlation here, maybe as hover effect to the table? We had this before in microbellm/templates/search_correlation.html so maybe you can re-use these elements
- [ ] add the same footer as in index

### Checkpoint 2: About page

### Checkpoint 3: update footer
- link to github should be https://github.com/GenomeNet/microbeLLM
- Contact should link to about -> contact section
- [ ] create a about page and privacy with dummy text
- [ ] should have a conteact section with my email philipp.muench@helmholtz-hzi.de


### Checkpoint 4: `phenotype_analysis.html`
- [ ] it should have the same page style as `artificial_dataset.html`, compeletly re-design the current content, transform all content according to style guide that it looks similar to /artificial_dataset page  or /knowledge_calibration
- [ ] it should use the right animation as we also show in the `index.html` on the box, in the same style as 
 `artificial_dataset.html` 
 - [ ] add the same footer as in index