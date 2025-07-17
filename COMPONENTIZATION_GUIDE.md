# Page Componentization Guide: From Monolithic to Modular

## Overview

This guide documents the process of transforming a single monolithic HTML page into a modular, component-based structure used in the microbeLLM project. We'll use the transformation of `knowledge_calibration_old.html` → `research/knowledge_calibration/` as a reference.

## Architecture Comparison

### Old Structure (Monolithic)
```
templates/
├── knowledge_calibration_old.html  # Single 2000+ line file
├── base.html                       # Base template
└── partials/                       # Shared components
```

### New Structure (Modular)
```
templates/
├── research_dynamic.html           # Dynamic page renderer
├── research/
│   └── knowledge_calibration/
│       ├── manifest.yaml          # Page configuration
│       └── sections/              # Individual components
│           ├── 00_hero_header.html
│           ├── 01_purpose_intro.html
│           ├── 02_definition_binomial.html
│           └── ... (40+ section files)
```

## Step-by-Step Transformation Process

### Step 1: Analyze the Monolithic Page

1. **Identify Major Sections**
   - Hero header with animation
   - Introduction/purpose text
   - Callout boxes (definitions, examples)
   - Interactive visualizations
   - Data tables
   - Footer

2. **Categorize Content Types**
   - Static text content
   - Interactive JavaScript components
   - CSS animations
   - Data-driven visualizations

### Step 2: Create Manifest Structure

Create `manifest.yaml` to define the page structure:

```yaml
page_config:
  title: "Hallucination Test"
  color_theme: "purple"  # Must match research_config.py
  project_id: "knowledge_calibration"

sections:
  - id: "hero_header"
    type: "hero"
    file: "sections/00_hero_header.html"
    title: "Hero Header"
    
  - id: "purpose_intro"
    type: "article"
    file: "sections/01_purpose_intro.html"
    title: "Hallucination Check: Fictional Strain Names"
    section_id: "purpose"
    include_sidebar: true
```

### Step 3: Extract Components

#### 3.1 Hero Header
Extract the hero section including:
- Title, author, subtitle
- Canvas animation setup
- Hero-specific styles

```html
<!-- 00_hero_header.html -->
<div class="hero-header hero-header--{{ project.color_theme }}">
    <canvas id="hero-canvas" class="hero-canvas"></canvas>
    <div class="hero-content">
        <h1 class="hero-title">{{ project.hero_title }}</h1>
        <div class="hero-author hero-author--{{ project.color_theme }}">{{ project.author }}</div>
        <p class="hero-subtitle">{{ project.hero_subtitle }}</p>
    </div>
</div>
```

#### 3.2 Article Sections
Extract article content, preserving:
- Paragraph structure
- Section headers
- Inline styles

```html
<!-- 01_purpose_intro.html -->
<p>Microbial information—like whether Pseudomonas putida degrades specific chemicals...</p>
<p>In practice, microbiologists gather strain-level details...</p>
```

#### 3.3 Callout Boxes
Extract inline callouts with proper type classification:

```html
<!-- 02_definition_binomial.html -->
<strong>Binomial names:</strong> Scientific names follow the format <em>Genus species</em>...
```

#### 3.4 Interactive Components
Extract JavaScript-driven components:

```html
<!-- 17_dynamic_stats.html -->
<div id="dynamicStatsText" class="article-text">
    <div class="loading-container">
        <div class="loading-progress">
            <div class="loading-progress-bar"></div>
        </div>
        <div class="loading-text">Loading statistics...</div>
    </div>
</div>

<script>
(function() {
    // Self-contained component logic
    loadDynamicStats();
})();
</script>
```

### Step 4: Handle Section Types

Each section type requires different handling:

#### Article Sections
- Standard article text
- May include sidebar
- Use `article` or `article_content` type

#### Callout Sections
- **Inline callouts**: Use `callout_inline` for indented boxes
- **Full-width callouts**: Use `section_callout` for gradient backgrounds
- **Dynamic callouts**: Use `section_callout_dynamic` for JS content

#### Special Sections
- **Hero**: Always first, includes animation
- **Raw**: Direct include without wrapper

### Step 5: Component Guidelines

#### 5.1 Self-Contained Components
Each component should be self-contained:
- Include all necessary HTML
- Inline styles specific to the component
- Self-executing JavaScript wrapped in IIFE

#### 5.2 Dynamic Components
For JavaScript-driven components:
```javascript
(function() {
    // Check if already initialized
    if (window._componentInitialized) return;
    window._componentInitialized = true;
    
    // Component logic here
    function initComponent() {
        // ...
    }
    
    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initComponent);
    } else {
        setTimeout(initComponent, 100);
    }
})();
```

#### 5.3 API Integration
Components fetching data should:
- Handle loading states
- Provide error feedback
- Work in component viewer with mock data

### Step 6: Create Component Viewer Entry

Add component to the viewer for isolated testing:

```python
# In web_app.py component routes
available_components = {
    'knowledge_calibration': [
        {'id': 'dynamic_stats', 'name': 'Dynamic Statistics'},
        {'id': 'knowledge_analysis_content', 'name': 'Knowledge Analysis Tables'},
        # ...
    ]
}
```

### Step 7: Test and Validate

1. **Full Page Testing**
   - Load `/research/knowledge_calibration/dynamic`
   - Verify all sections render correctly
   - Check animations and interactions

2. **Component Testing**
   - Load individual components in `/components/knowledge_calibration/[component_id]`
   - Verify standalone functionality
   - Test with mock data

## Applying to Phenotype Analysis Page

To transform `phenotype_analysis.html`:

### 1. Create Directory Structure
```bash
mkdir -p templates/research/phenotype_analysis/sections
```

### 2. Create Manifest
```yaml
# templates/research/phenotype_analysis/manifest.yaml
page_config:
  title: "Phenotype Analysis"
  color_theme: "green"
  project_id: "phenotype_analysis"

sections:
  - id: "hero_header"
    type: "hero"
    file: "sections/00_hero_header.html"
    
  - id: "intro"
    type: "article"
    file: "sections/01_intro.html"
    title: "Understanding Phenotype Predictions"
    include_sidebar: true
```

### 3. Extract Sections

#### Hero Section
- Extract hero with bacteria animation
- Use green color theme

#### Key Phenotype Categories
- Extract as `section_callout--green`
- Include overview cards

#### Analysis Methodology
- Extract as article section
- Include methodology callout

#### Model Predictions
- Extract as `section_callout_dynamic`
- Include loading states and data tables

### 4. Component Extraction Checklist

- [ ] Hero header with animation
- [ ] Introduction text sections
- [ ] Phenotype categories overview
- [ ] Interactive phenotype animation
- [ ] Model analysis tables
- [ ] Hidden phenotypes section
- [ ] Dynamic statistics
- [ ] Consensus analysis (if present)

### 5. JavaScript Migration

Extract and modularize:
- Phenotype animation functions
- Data loading logic
- Chart rendering
- Tooltip handlers

### 6. Style Considerations

- Use green color theme throughout
- Maintain consistent spacing variables
- Preserve responsive design

## Common Pitfalls and Solutions

### 1. JavaScript Conflicts
**Problem**: Multiple instances of same component
**Solution**: Use instance IDs and namespace checking

### 2. CSS Specificity
**Problem**: Styles bleeding between components
**Solution**: Scope styles within component containers

### 3. Data Dependencies
**Problem**: Components expecting global data
**Solution**: Make components fetch their own data

### 4. Animation Performance
**Problem**: Multiple animations causing lag
**Solution**: One animation per page, in hero only

## Benefits of Componentization

1. **Maintainability**: Update individual sections without touching others
2. **Reusability**: Share components across pages
3. **Testing**: Test components in isolation
4. **Performance**: Load only needed components
5. **Collaboration**: Multiple developers can work on different sections

## Next Steps for Phenotype Analysis

1. Create the manifest file
2. Extract hero section with bacteria animation
3. Split article content into logical sections
4. Extract interactive visualizations
5. Create component viewer entries
6. Test both full page and components
7. Update research_config.py with page metadata

This componentization approach ensures consistent structure across all research pages while maintaining flexibility for unique content and interactions.