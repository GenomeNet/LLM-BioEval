# Standard Workflow

First, think through the problem, read the codebase for relevant files, and write a plan to [PROJECTPLAN.md](PROJECTPLAN.md).
The plan should have a list of todo items that you can check off as you complete them.
Before you begin working, check in with me and I will verify the plan.
Then, begin working on the todo items, marking them as complete as you go.
Please, at every step of the way, just give me a high-level explanation of what changes you made.
Make every task and code change as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
Finally, add a review section to the [PROJECTPLAN.md](PROJECTPLAN.md) file with a summary of the changes you made and any other relevant information.


## Style guide

### General Principles
- All pages should use a consistent, modern layout as seen in `index.html` and `research.html`.
- The default background for all main content sections is white (`background: var(--bg-primary)`), providing a clean and minimal look.
- Only callout boxes should have colored or gradient backgrounds; all other sections remain white.
- All pages are fully responsive, with grid and flex layouts adapting to smaller screens.
- Animations (e.g., canvas-based bacteria, DNA, or cell growth) are used for visual interest but do not interfere with content readability.

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
- **Base Callout** (`.callout`): White background, indented 120px from left, with border and shadow
- **Definition Callout** (`.callout-definition`): Purple gradient background for definitions
- **Template Callout** (`.callout-template`): Alternative purple gradient
- **Phenotype Callout** (`.callout-phenotype`): Green gradient for phenotype content
- **Methodology Callout** (`.callout-methodology`): Green-to-yellow gradient

#### Full-Width Sections
- **Wide Section** (`.section-wide`): Full viewport width with white background
- **Callout Section** (`.section-callout`): Full viewport width with gradient background
  - Purple variant: `.section-callout--purple` for knowledge/hallucination themes
  - Green variant: `.section-callout--green` for phenotype themes
- **Content Wrapper**: Use `.section-wide__content` or `.section-callout__content` inside

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
- **Table** (`.table`): Base table with gray header and hover states
- **Table Title** (`.table-title`): 16px title above tables

#### Paper Info Section
Consistent paper/article metadata display:
- **Paper Info Card** (`.paper-info-card`): White card with border, like TOC
- **Paper Info List** (`.paper-info-list`): Styled list with dividers
- List items show metadata with subtle styling

#### Figures and Charts
- **Figure** (`.figure`): Container with consistent spacing
- **Figure Title** (`.figure-title`): 16px bold title
- **Figure Caption** (`.figure-caption`): 14px gray caption text
- **Chart Container** (`.chart-container`): White bordered container for interactive elements

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
    <!-- Full-width callout content -->
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
