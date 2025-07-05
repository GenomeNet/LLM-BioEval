# Standard Workflow

First, think through the problem, read the codebase for relevant files, and write a plan to [PROJECTPLAN.md](PROJECTPLAN.md).
The plan should have a list of todo items that you can check off as you complete them. Before you begin working, check in with me and I will verify the plan. Then, begin working on the todo items, marking them as complete as you go. Please, at every step of the way, just give me a high-level explanation of what changes you made. Make every task and code change as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity. Finally, add a review section to the [PROJECTPLAN.md](PROJECTPLAN.md) file with a summary of the changes you made and any other relevant information.

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

## Style guide

### General Principles
- All pages should use a consistent, modern layout as seen in `index.html` and `research.html`.
- The default background for all main content sections is white (`background: var(--bg-primary)`), providing a clean and minimal look.
- Only callout boxes should have colored or gradient backgrounds; all other sections remain white.
- Each research project has a defined color theme (`purple`, `green`, etc.) in `research_config.py` - all callout sections within a project page MUST use only that project's color theme.
- All pages are fully responsive, with grid and flex layouts adapting to smaller screens.
- Animations (e.g., canvas-based bacteria, DNA, or cell growth) are used for visual interest but do not interfere with content readability.

### Content Organization Guidelines

#### What Goes Where
1. **Standard Article Sections** (white background):
   - Main text content, paragraphs, and explanations
   - Regular headings and subheadings
   - Inline images and simple figures
   - Basic lists and text-based content

2. **Indented Callout Boxes** (`.callout`):
   - Definitions and terminology explanations
   - Small animations or interactive examples
   - Supplementary information and asides
   - Notes, tips, or warnings
   - Brief summaries (e.g., "200 total transformations")

3. **Full-Width Callout Sections** (`.section-callout`):
   - Major results and key findings (e.g., "Top Performing Models")
   - Performance showcases and rankings
   - Important visualizations that deserve emphasis
   - Hero-style content blocks with gradient backgrounds

4. **Full-Width Standard Sections** (`.section-wide`):
   - Large tables that need horizontal space
   - Complex data visualizations
   - Multi-column layouts
   - Content that benefits from extra width but doesn't need emphasis

#### Text Width Rules
- Article text should always be constrained to 850px maximum width for readability
- Even in full-width sections, wrap text content in a container with `max-width: 850px`
- Only tables, charts, and visualizations should extend beyond this width
- Use `margin: 0 auto` to center constrained content within wide sections

### Spacing System
Spacing between sections and components is managed with CSS variables for consistent vertical rhythm:
- `--spacing-section: 48px` - Between major sections
- `--spacing-component: 32px` - Between components within sections
- `--spacing-element: 24px` - Between elements within components
- `--spacing-small: 16px` - For smaller gaps and padding
- Use calc() for fractional spacing: `calc(var(--spacing-small) * 0.5)` for 8px, etc.
- Avoid hardcoded px values; always use CSS variables or calc() expressions

### Article Components (from article_styles.css)

#### Text Styles
- **Main Article Text** (`.article-text`): 18px font size, line-height 1.7, constrained to 850px width in article containers
- **Full-width Text**: Use `.article-text` inside `.section-wide` or `.section-callout` for full-width text
- **Paragraph Spacing**: Paragraphs have `calc(var(--spacing-element) * 0.833)` bottom margin

#### Title Hierarchy
- **Main Section Title** (`.main-section-title`): 48px, bold 800, with bottom border
- **Section Title** (`.section-title`): 32px, bold 700
- **Section Header** (`.section-header`): Flex layout with uppercase 14px text for subtitles

#### Callout Boxes
Indented boxes for special content with consistent styling:
- **Base Callout** (`.callout`): Off-white background (#FAFAFA), indented 120px from left, with border and shadow
- **Definition Callout** (`.callout-definition`): Purple gradient background for definitions
- **Template Callout** (`.callout-template`): Alternative purple gradient
- **Phenotype Callout** (`.callout-phenotype`): Green gradient for phenotype content
- **Methodology Callout** (`.callout-methodology`): Green-to-yellow gradient

**IMPORTANT: Color Theme Consistency Rules**
- Each research project has a defined color theme in `research_config.py`
- All callout sections within a project page MUST use only that project's color theme
- Knowledge calibration pages: Use ONLY purple variants (`.section-callout--purple`, `.callout-definition`, `.callout-template`)
- Phenotype analysis pages: Use ONLY green variants (`.section-callout--green`, `.callout-phenotype`, `.callout-methodology`)
- Never mix color themes within a single page

#### Full-Width Sections
- **Wide Section** (`.section-wide`): Full viewport width with white background
- **Callout Section** (`.section-callout`): Full viewport width with gradient background
  - Purple variant: `.section-callout--purple` for knowledge/hallucination themes
  - Green variant: `.section-callout--green` for phenotype themes
  - **Color theme MUST match** the project's `color_theme` setting in `research_config.py`
- **Content Wrapper**: Use `.section-wide__content` or `.section-callout__content` inside
- **Section Header Structure**: All section-callout boxes should include:
  - `.section-callout__header` container
  - `.section-callout__title` for the main heading (24px, bold)
  - `.section-callout__text` for descriptive text (18px, normal)
- **Dynamic Content Rule**: All dynamic/interactive content (charts, visualizations, tables) should be placed in callout sections rather than standard article sections
- **Full-Width Content**: For wide tables and data that need full page width, use `.section-callout__content--full-width` instead of `.section-callout__content`

#### Bar Charts
Consistent bar chart styling with three levels of rounding:
- **Standard Bars** (`.bar-chart-container`): 12px border-radius for table bars
- **Template Bars** (`.template-bar-wrapper`): 4px border-radius for distribution bars
- **Correlation Bars** (`.correlation-bar`): 2px border-radius for minimal style
- **Bar Segments**: Use `.bar-segment` with color classes: `.na`, `.limited`, `.moderate`, `.extensive`

#### Legends
Flexible legend system with variants:
- **Base Legend** (`.legend`): Left-aligned by default, wraps on small screens
- **Centered Legend** (`.legend--centered`): Center-aligned variant
- **Boxed Legend** (`.legend--boxed`): Gray background with padding
- **Separated Legend** (`.legend--separated`): Top border separator
- **Legend Items**: Use `.legend-item` with `.legend-color` for color swatches

#### Tables
Modern responsive tables:
- **Table Wrapper** (`.table-responsive`): Scrollable container with border
- **Table** (`.table`): Base table with off-white background (#FAFAFA) and headers
- **Table Title** (`.table-title`): 16px title above tables
- **Tables in Callouts**: When placing tables inside `.section-callout`, borders and shadows are automatically removed to maintain clean integration with the gradient background

#### Paper Info Section
Consistent paper/article metadata display:
- **Paper Info Card** (`.paper-info-card`): Off-white card (#FAFAFA) with border, like TOC
- **Paper Info List** (`.paper-info-list`): Styled list with dividers
- List items show metadata with subtle styling

#### Figures and Charts
- **Figure** (`.figure`): Container with consistent spacing
- **Figure Title** (`.figure-title`): 16px bold title
- **Figure Caption** (`.figure-caption`): 14px gray caption text
- **Chart Container** (`.chart-container`): Off-white background (#FAFAFA) container for interactive elements; avoid borders on chart areas to maintain clean visual separation

#### Color Guidelines for Boxes and Containers
- **Concept Boxes**: Use off-white (#FAFAFA) instead of pure white for animations, examples, and explanatory content
- **Table Backgrounds**: Use off-white (#FAFAFA) for table body and header backgrounds
- **Interactive Elements**: Use off-white (#FAFAFA) for chart containers and other interactive components

### Usage Examples

#### Article Page Structure
```html
<article class="article-container">
  <div class="article-content">
    <h1 class="main-section-title">Article Title</h1>
    <p class="article-text">Article content...</p>
    
    <div class="callout callout-definition">
      <div class="callout-content">Special content...</div>
    </div>
  </div>
</article>

<section class="section-callout section-callout--purple">
  <div class="section-callout__content">
    <div class="section-callout__header">
      <h3 class="section-callout__title">Section Title</h3>
      <p class="section-callout__text">Descriptive text explaining the purpose and content of this section.</p>
    </div>
    <!-- Main callout content -->
  </div>
</section>
```

#### Table with Bar Charts
```html
<div class="table-responsive">
  <table class="table">
    <thead>...</thead>
    <tbody>
      <tr>
        <td>
          <div class="bar-chart-container">
            <div class="bar-chart-segments">
              <div class="bar-segment na" style="width: 30%">30%</div>
              <div class="bar-segment moderate" style="width: 70%">70%</div>
            </div>
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</div>
```

#### Legend Examples
```html
<!-- Left-aligned legend -->
<div class="legend">
  <div class="legend-item">
    <div class="legend-color" style="background: #e5e7eb"></div>
    <span>No Information</span>
  </div>
</div>

<!-- Centered legend with background -->
<div class="legend legend--centered legend--boxed">
  <!-- Legend items -->
</div>
```

### Other UI Elements
- Publication and project cards use white background, 16px rounded corners, and subtle borders
- Section headers use flex layout, left-aligned, with small uppercase text
- Buttons and links use consistent styling, with primary actions in color
- Footer uses light gray border and background
