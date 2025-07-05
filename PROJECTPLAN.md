# Project plan 

## Project overview 

MicrobeLLM is a Python tool designed to evaluate Large Language Models (LLMs) on their ability to predict microbial phenotypes. This tool helps researchers assess how well different LLMs perform on microbiological tasks by querying them with bacterial species names and comparing their predictions against known characteristics.

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
- [x] correct file paths, "Testing LLM knowledge on real and artificial bacterial species" box on `index.html` links to /artificial_dataset but this will display code from `microbellm/templates/knowledge_calibration.html`. So we should rename the url path /artificial_dataset to /knowledge_calibration and update `/research` and `/index` to point to the right paths
- [x] consistent headlines, it should have the structure, h1 on "Hallucination Check: Fictional Strain Names" and "Web-Aligned Knowledge: Real Bacterial Names vs. Google Counts", this should be maybe visible with a grey line and background gradient (to white) like we had defined before. H2 should be "Top Performing Models" and "Full Results of the Artificial Hallucination Test" (maybe rename it to be more consistent with content). Again, H2 should be "Top Models – Web-Alignment" and "Knowledge-Web Alignment Table" 
- [x] all full tables should by default not be shown in total, this is currently done for tables under "Full Results of the Artificial Hallucination Test" but not for "Knowledge-Web Alignment Table" 
- [x] "Top Performing Models" boxes should have the same style, they are currently different for the 2 types 
- [x] the only indentation should be for "callout boxes" and not for parts of the content. There is indentation that should be removed for "Top Performing Models" 
- [x] consistent indentation according to style guide, "template" example is not indented as other boxes, "Top Performing Models" is wrongfully indented since it's normal content and not a call-out 
- [x] add a bar plot of the model QUALITY SCORE, x should be the score and y shoudl be different models. This shoudl be displayed under "Full Results of the Artificial Hallucination Test
".  One bar per template showing all within boxes side by side with annotation we need to understand these. this should show the distribution of models
- [x] add a similar bar plot for "Knowledge-Web Alignment Table", this time we should show CORRELATION scroes. 
- [ ] since we show here correlation of google search results vs the konwlege group membership we can add a new plot showing the correlation here, maybe as hover effect to the table? We had this before in microbellm/templates/search_correlation.html so maybe you can re-use these elements
- [ ] add the same footer as in index


### Checkpoint 2: update footer
- [ ] link to github should be https://github.com/GenomeNet/microbeLLM
- [ ] Contact should link to about -> contact section
- [ ] create a about page and privacy with dummy text
- [ ] should have a conteact section with my email philipp.muench@helmholtz-hzi.de


### Checkpoint 3: `phenotype_analysis.html`
- [x] it should have the same page style as `artificial_dataset.html`, compeletly re-design the current content, transform all content according to style guide that it looks similar to /artificial_dataset page  or /knowledge_calibration
- [x] it should use the right animation as we also show in the `index.html` on the box, in the same style as 
 `artificial_dataset.html` 
 - [x] add the same footer as in index

### Current Task: Second Iteration - Agent B

**Task**: Complete second iteration improvements for phenotype_analysis.html based on user feedback

**Todo Items**:
- [x] Add author name "Münch et al. 2024" matching knowledge_calibration.html style
- [x] Add descriptive text sections explaining the analysis
- [x] Add callout boxes with phenotype and methodology explanations
- [x] Add animated phenotype categories display
- [x] Fix CSS/JavaScript corruption issues
- [x] Ensure tables match the styling from knowledge_calibration.html

**Files Modified**: `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm/templates/phenotype_analysis.html`

### Agent B Review - Second Iteration Completed

**Major Improvements Implemented**:
1. **Author Attribution**: Added "Münch et al. 2024" in green color (#10b981) matching the phenotype theme
2. **Content Structure**: 
   - Added comprehensive "Understanding Phenotype Predictions" section with descriptive text
   - Added "Analysis Methodology" section explaining the evaluation approach
3. **Callout Boxes**:
   - Implemented callout-phenotype with green gradient background for key categories
   - Implemented callout-methodology with teal gradient for analysis explanation
4. **Animated Elements**:
   - Created phenotype-categories-animated grid with 6 phenotype items
   - Added fadeInUp animation with staggered delays for visual appeal
   - Used appropriate icons for each phenotype category
5. **Styling Improvements**:
   - Added proper table styles matching knowledge_calibration.html
   - Fixed significant CSS/JavaScript corruption (removed ~920 lines of orphaned CSS)
   - Properly wrapped orphaned CSS in style tags
6. **Technical Fixes**:
   - Resolved template structure issues
   - Ensured all blocks are properly closed
   - Maintained consistent styling with other pages

## Review

### Checkpoint 1 Completed

**Major Tasks Completed**:
1. **URL Routing**: Updated Flask routing from `/artificial_dataset` to `/knowledge_calibration` and updated all navigation links
2. **Headline Structure**: Converted main section headers to h1 with grey line styling
3. **Table Collapsing**: Added expand/collapse functionality to Knowledge-Web Alignment Table
4. **Style Standardization**: Made both "Top Performing Models" sections use consistent styling
5. **Histogram Implementation**: Added Quality Score and Correlation Score distribution histograms

### Bug Fixes Implemented

**Quality Score Distribution Issues Fixed**:
1. **Bar Display**: 
   - Increased histogram height from 120px to 200px for better visibility
   - Added minimum bar height of 10% and padding for score labels
   - Improved bar styling with flexbox alignment
   - Limited display to top 20 models to prevent excessive width

2. **Correlation Histogram**:
   - Fixed scope issue by moving `generateCorrelationScoreHistogram` to global scope
   - Added missing data fields (knowledgeDist, speciesCount) to modelResults
   - Implemented same display improvements as Quality Score histogram

3. **CSS Improvements**:
   - Updated histogram-chart height across all instances
   - Added overflow handling for wide histograms
   - Improved label rotation for better readability
   - Added minimum widths for histogram bins

**Files Modified**: `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm/templates/knowledge_calibration.html`

### Additional Improvements

**Quality Score Calculation Fix**:
1. **Fixed Template Filtering**: Updated the overallModelScores calculation to only use templates from templateArray instead of ALL templates in the data
2. **Corrected Model Ranking**: The "Model Quality Score Distribution" chart now shows accurate scores based only on the selected templates
3. **Removed Duplicate**: Cleared the duplicate histogram container to avoid confusion
4. **Improved Accuracy**: Rankings and scores now match the data shown in the detailed tables below

### Checkpoint 3 Completed - Agent B

**Major Tasks Completed**:
1. **Page Redesign**: Completely redesigned phenotype_analysis.html to match the style guide from knowledge_calibration.html
2. **Hero Section**: Added hero header with gradient background and animation canvas
3. **Content Structure**: Implemented proper layout with overview cards section and model analysis section
4. **Animation**: Added bacteria animation adapted from index.html with green color scheme
5. **Footer**: Added consistent footer matching index.html with updated GitHub link

**Implementation Details**:
1. **Hero Section**:
   - Green gradient background matching phenotype theme
   - Animated bacteria canvas in header
   - Clear title and subtitle explaining the analysis

2. **Overview Cards**:
   - Full-width callout section with gradient background
   - Three information cards explaining analysis features
   - Icons and hover effects matching style guide

3. **Technical Challenges**:
   - File had significant CSS/JavaScript corruption from previous edits
   - Mixed CSS fragments throughout JavaScript code
   - Successfully preserved all original JavaScript functionality
   - Added animation initialization on page load

**Files Modified**: `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm/templates/phenotype_analysis.html`

### Current Task: Update createQualityScoreBarChart to Show All Templates

**Problem**: The `createQualityScoreBarChart` function currently shows average quality scores across all templates. Need to update it to show individual scores for each template instead of just averaging them.

**Analysis**: 
- Current function (line ~7691) takes sorted model scores with averageQualityScore
- Need to modify to show scores broken down by template
- Data structure has scores available per template in the modelData object

**Todo Items**:
- [ ] Modify createQualityScoreBarChart to accept template-specific data
- [ ] Create grouped bar chart showing scores for each template
- [ ] Update the function call to pass template information
- [ ] Add template legend/labels to the chart
- [ ] Test changes to ensure proper visualization

**Files to Modify**: `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm/templates/knowledge_calibration.html`

### Spacing Standardization Review

**Major Tasks Completed**:
1. **Knowledge Calibration HTML**:
   - Replaced all hardcoded margin-bottom values with CSS variables
   - Replaced all hardcoded padding values with CSS variables
   - Replaced all hardcoded gap values with CSS variables
   - Standardized responsive media query spacing values

2. **Phenotype Analysis HTML**:
   - Updated hero section spacing to use CSS variables
   - Standardized overview cards and phenotype item spacing
   - Replaced hardcoded padding and margins with CSS variables

3. **Style Guide Update**:
   - Added comprehensive spacing documentation to CLAUDE.md
   - Documented the four main spacing variables:
     - `--spacing-section: 48px` for major sections
     - `--spacing-component: 32px` for components
     - `--spacing-element: 24px` for elements
     - `--spacing-small: 16px` for small gaps
   - Added guidance for using calc() for fractional spacing

**Key Improvements**:
- Consistent spacing throughout both templates
- Easier maintenance through centralized CSS variables
- Responsive designs maintain spacing proportions
- Eliminated over 100 hardcoded px values
- Standardized spacing patterns across all components

**Files Modified**: 
- `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm/templates/knowledge_calibration.html`
- `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm/templates/phenotype_analysis.html`
- `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/CLAUDE.md`

### Article Styles Standardization Review

**Major Tasks Completed**:
1. **Created article_styles.css**:
   - Comprehensive CSS file with reusable article components
   - Includes styles for text, titles, callout boxes, and full-width sections
   - Supports both purple (knowledge) and green (phenotype) color themes
   - Responsive design with proper breakpoints

2. **Updated knowledge_calibration.html**:
   - Added import for article_styles.css
   - Removed duplicate styles for article text, titles, sections, and callouts
   - Updated HTML to use new classes (hero-author--purple, section-callout--purple, section-wide)
   - Cleaned up empty CSS rulesets
   - Maintained all unique page-specific styles

3. **Updated phenotype_analysis.html**:
   - Added import for article_styles.css
   - Removed duplicate styles for article components
   - Updated HTML to use new classes (hero-author--green)
   - Cleaned up duplicate callout definitions
   - Fixed methodology callout gradient in article_styles.css

**Key Components in article_styles.css**:
- **Article Text**: `.article-text` with proper width constraints
- **Titles**: `.main-section-title`, `.section-title`, `.section-header`
- **Callout Boxes**: `.callout` with variants for different themes
- **Full-Width Sections**: `.section-wide`, `.section-callout` with color variants
- **Utility Classes**: Author attribution, content width helpers

**Benefits Achieved**:
- Consistent styling across all article pages
- Reduced code duplication (~300 lines removed)
- Easier maintenance through centralized styles
- Clear naming conventions following BEM-like patterns
- Preserved all functionality while improving organization

**Files Modified**: 
- Created: `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm/static/css/article_styles.css`
- Updated: `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm/templates/knowledge_calibration.html`
- Updated: `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm/templates/phenotype_analysis.html`

### Article Styles Enhancement Review

**Major Tasks Completed**:
1. **Enhanced article_styles.css with new components**:
   - Added standardized bar chart styles with three levels of rounding (12px, 4px, 2px)
   - Added flexible legend system with variants (base, centered, boxed, separated)
   - Added modern table styles with responsive wrapper
   - Added paper info section styles matching TOC design
   - Added figure and chart container styles

2. **Bar Chart Standardization**:
   - `.bar-chart-container`: 12px radius for main table bars
   - `.template-bar-wrapper`: 4px radius for distribution bars
   - `.correlation-bar`: 2px radius for minimal style
   - Consistent color classes for segments (na, limited, moderate, extensive)

3. **Legend System**:
   - Base legend is left-aligned with proper wrapping
   - Centered variant for symmetrical layouts
   - Boxed variant with gray background
   - Separated variant with top border
   - Consistent legend item and color swatch styling

4. **Updated CLAUDE.md with comprehensive style guide**:
   - Reorganized into clear sections (General Principles, Spacing, Components)
   - Added detailed documentation for all article components
   - Included usage examples with HTML snippets
   - Clear hierarchy and naming conventions

**Key Improvements**:
- Resolved bar chart rounding inconsistencies
- Standardized legend alignment (left by default, centered as option)
- Fixed legend background inconsistencies (white by default, gray as option)
- Created reusable paper info section component
- Comprehensive documentation for future development

**Files Modified**: 
- Updated: `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/microbellm/static/css/article_styles.css`
- Updated: `/Users/pmu15/Documents/github.com/GenomeNet/microbeLLM/CLAUDE.md`