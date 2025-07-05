# Project plan 

## Project overview 

MicrobeLLM is a Python tool designed to evaluate Large Language Models (LLMs) on their ability to predict microbial phenotypes. This tool helps researchers assess how well different LLMs perform on microbiological tasks by querying them with bacterial species names and comparing their predictions against known characteristics.

## TODO
- [x] add dummy text between "Top Performing Models" and "Models Needing Improvement" (both .section-callout`). The dummy text should be normal Standard Article Sections
- [x] table background at "template1_knowlege" and so on should be the grey color FAFAFA, sometimes its opaque which looks odd
- [x] this section "Web-Aligned Knowledge: Real Bacterial Names vs. Google Counts" and the text below that should be Standard Article Section, currently its over the full page 
- [x] add dummy text after Top Models – Web-Alignment and before "Correlation Visualization: Google Search vs. Knowledge Level" , again in "Standard Article Sections" style 
- [x] render the legend in "Correlation Visualization: Google Search vs. Knowledge Level" more like we render the legends in "Knowledge-Web Alignment Process". 
- [x] plot in "Correlation Visualization: Google Search vs. Knowledge Level" should have a grey background #FAFAFA  

## Review

### Summary of Changes Made

All TODO items have been successfully completed. Here's what was changed in `knowledge_calibration.html`:

1. **Added dummy text between sections** - Added appropriate article text between "Top Performing Models" and "Models Needing Improvement" sections to provide context and improve content flow.

2. **Fixed table backgrounds** - Added CSS to ensure tables in section-callout sections maintain the #FAFAFA background color instead of becoming transparent.

3. **Converted section to Standard Article** - Changed "Web-Aligned Knowledge: Real Bacterial Names vs. Google Counts" from a full-width section to a standard article section for consistency.

4. **Added transitional content** - Inserted article text between "Top Models – Web-Alignment" and "Correlation Visualization" sections to improve narrative flow.

5. **Updated legend styling** - Modified the legend in "Correlation Visualization" to use the same `flow-legend` class as the "Knowledge-Web Alignment Process" section for visual consistency.

6. **Applied grey backgrounds** - Updated all scatter plots and charts in the correlation visualization to use #FAFAFA background instead of white, including both canvas and SVG elements.

### Technical Notes

- All changes follow the established style guide in CLAUDE.md
- Used existing CSS classes and patterns for consistency
- Maintained responsive design principles
- No new dependencies or complex changes were introduced
- All modifications were minimal and focused on the specific requirements

## Refactor Research Project Content (New Task)

### Problem
Research project information (titles, subtitles, authors, DOIs, color schemes) was duplicated across multiple templates:
- index.html
- research.html  
- Individual research pages (knowledge_calibration.html, phenotype_analysis.html)

### Solution
Create a centralized configuration system to manage all research project metadata in one place.

### Implementation Steps

- [x] Create centralized research projects configuration file
- [x] Extract research metadata from existing templates
- [x] Modify web_app.py to load and serve research configuration data
- [x] Update research.html template to use dynamic data
- [x] Update index.html template to use dynamic data
- [x] Update individual research pages to use dynamic metadata

### Review

#### Changes Made

1. **Created `microbellm/research_config.py`**
   - Defined a `ResearchProject` dataclass to store project metadata
   - Created a list of all research projects with their properties
   - Added helper functions to retrieve projects by ID, route, or page

2. **Updated `microbellm/web_app.py`**
   - Imported the research configuration module
   - Modified routes to pass project data to templates:
     - `/` (index) - passes homepage projects
     - `/research` - passes all projects
     - `/knowledge_calibration` - passes specific project data
     - `/phenotype_analysis` - passes specific project data

3. **Updated Templates to Use Dynamic Data**
   - **research.html**: Replaced hardcoded project cards with dynamic loop
   - **index.html**: Replaced hardcoded publication cards with dynamic loop
   - **knowledge_calibration.html**: Updated hero section and paper info to use project data
   - **phenotype_analysis.html**: Updated hero section to use project data

4. **Improved Animation Handling**
   - Updated animation functions to accept canvas ID parameters
   - Made animation initialization dynamic based on projects displayed

#### Benefits

1. **Single Source of Truth**: All research metadata is now centralized in one configuration file
2. **Easy Updates**: Adding new research projects or updating existing ones requires changes in only one place
3. **Consistency**: Ensures all pages display the same information for each project
4. **Maintainability**: Reduces code duplication and makes the codebase easier to maintain
5. **Extensibility**: Easy to add new properties to projects (e.g., tags, categories, related links)

#### Future Enhancements

1. **Database Storage**: Consider moving project data to the SQLite database for even more flexibility
2. **Admin Interface**: Add web interface for managing research projects without code changes
3. **Search/Filter**: Add functionality to search and filter research projects
4. **Project Pages**: Create a dynamic route handler for all project pages instead of individual routes
5. **Markdown Support**: Allow project descriptions to use Markdown for richer formatting

## Color Theme Consistency Fix

### Problem
Callout sections were using incorrect color themes that didn't match the project configuration:
- knowledge_calibration.html had green callouts (should be purple)
- phenotype_analysis.html had purple callouts (should be green)

### Solution
1. Fixed all callout sections to use the correct color themes based on `research_config.py`
2. Updated CLAUDE.md style guide to enforce color theme consistency rules

### Changes Made
1. **knowledge_calibration.html**: Changed all `section-callout--green` to `section-callout--purple`
2. **phenotype_analysis.html**: Changed all `section-callout--purple` to `section-callout--green`
3. **CLAUDE.md**: Added explicit rules about color theme consistency:
   - Each project must only use its defined color theme
   - Knowledge pages: purple variants only
   - Phenotype pages: green variants only
   - Never mix color themes within a single page

### Note
When the growth_conditions project is implemented, a `.section-callout--yellow` variant will need to be added to `article_styles.css` to match its yellow color theme.

## Hero Header Refactoring

### Problem
The hero header implementation was duplicated across research pages:
- Each template had its own copy of the CSS with hardcoded colors
- The HTML structure was repeated in each template
- Although project data was used, the implementation was inconsistent

### Solution
Created a reusable hero header component using Flask template partials.

### Implementation Steps
- [x] Create partials directory and hero_header.html template
- [x] Move shared hero header CSS to article_styles.css with dynamic color classes
- [x] Update knowledge_calibration.html to use the partial
- [x] Update phenotype_analysis.html to use the partial  
- [x] Test both pages to ensure proper rendering

### Changes Made

1. **Created `/templates/partials/hero_header.html`**
   - Reusable template partial for hero headers
   - Dynamic canvas animation based on project color theme
   - Particle-based animation with theme-specific colors and behaviors

2. **Updated `article_styles.css`**
   - Added complete hero header styles
   - Created theme-specific classes: `.hero-header--purple`, `.hero-header--green`, `.hero-header--yellow`
   - Dynamic color gradients and animations for each theme
   - Responsive styles for mobile devices

3. **Updated Templates**
   - **knowledge_calibration.html**: Removed 85 lines of duplicate CSS, replaced hero HTML with `{% include 'partials/hero_header.html' %}`
   - **phenotype_analysis.html**: Removed 83 lines of duplicate CSS, replaced hero HTML with include statement

### Benefits

1. **DRY Principle**: Eliminated code duplication across templates
2. **Consistency**: All hero headers now use identical structure and styling
3. **Dynamic Theming**: Colors automatically match project configuration
4. **Easy Maintenance**: Single location for hero header updates
5. **Future-Ready**: Easy to add new color themes (e.g., yellow for growth_conditions)
6. **Better Performance**: Shared CSS loaded once instead of per-page

### Technical Details

The hero header now:
- Uses the project's `color_theme` to determine colors
- Generates a unique canvas ID based on project ID
- Applies theme-specific gradients and particle animations
- Properly displays author information with theme-matched styling

## Footer Refactoring

### Problem
Footer implementation was duplicated across templates with inconsistent links and styles:
- index.html had footer with all links including Imprint
- phenotype_analysis.html had footer missing Imprint link
- Other pages (knowledge_calibration, research, about, imprint, privacy) had no footers
- Footer CSS was duplicated in individual templates

### Solution
Created a reusable footer partial similar to the hero header implementation.

### Implementation Steps
- [x] Analyze footer implementations across templates
- [x] Create footer partial template
- [x] Move footer CSS to a centralized location
- [x] Update all templates to use the footer partial
- [x] Test all pages to ensure proper footer rendering

### Changes Made

1. **Created `/templates/partials/footer.html`**
   - Standardized footer with all links (About, Contact, GitHub, Imprint, Privacy)
   - Consistent copyright text: "© 2025 - LLM-BioEval Team"

2. **Added footer styles to `article_styles.css`**
   - Moved all footer CSS from individual templates
   - Added responsive styles for mobile devices
   - Ensures consistent styling across all pages

3. **Updated all templates**
   - **index.html**: Removed footer CSS, replaced footer HTML with `{% include 'partials/footer.html' %}`
   - **phenotype_analysis.html**: Removed footer CSS and inconsistent footer, replaced with partial
   - **knowledge_calibration.html**: Added footer include
   - **research.html**: Added footer include after projects section
   - **about.html**: Added footer include
   - **imprint.html**: Added footer include
   - **privacy.html**: Added footer include

### Benefits

1. **Consistency**: All pages now have identical footers with the same links
2. **Maintainability**: Single location for footer updates
3. **Complete Coverage**: All main pages now have footers
4. **DRY Principle**: Eliminated code duplication
5. **Responsive Design**: Footer adapts properly on mobile devices