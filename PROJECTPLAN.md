# Project plan 

## Project overview 

MicrobeLLM is a Python tool designed to evaluate Large Language Models (LLMs) on their ability to predict microbial phenotypes. This tool helps researchers assess how well different LLMs perform on microbiological tasks by querying them with bacterial species names and comparing their predictions against known characteristics.

## CSS Architecture Refactoring and Generic Template Creation (Completed 2025-07-06)

### Summary
Successfully implemented a three-layer CSS architecture and created a generic research template system:

1. **CSS Architecture**:
   - Created `component_styles.css` for visual styling only
   - Created `layout_styles.css` for positioning and spacing
   - Updated wrapper templates with `in_viewer` parameter
   - Component viewer now shows pure components without layout

2. **Generic Research Template**:
   - Created `research_article.html` as reusable template
   - Updated routing in `web_app.py` to use generic template
   - Fixed full-width callout display issues in component viewer

3. **Current Status**:
   - ✅ Three-layer CSS architecture implemented
   - ✅ Generic research template functional
   - ✅ Component viewer displaying components correctly
   - ✅ Proper separation of concerns achieved

### Next Steps:
- Add missing JavaScript components from old template
- Update CLAUDE.md documentation with new architecture details

## Article Layout Improvements (Completed 2025-07-06)

### Summary
Updated the modular article system to improve alignment and appearance to match the old template style:

1. **Template Simplification**:
   - Simplified `research_article.html` wrapper structure
   - Removed unnecessary layout wrapper divs
   - Direct rendering of callout classes without wrapper partials
   - Cleaner section callout rendering

2. **CSS Updates**:
   - **article_styles.css**: Fixed spacing, alignment, and callout indentation (120px)
   - **layout_styles.css**: Removed redundant layout wrappers and conflicting styles
   - Moved article-specific styles from layout to article CSS for consistency

3. **Results**:
   - ✅ Cleaner HTML structure with less nesting
   - ✅ Proper 120px left indentation for callout boxes
   - ✅ Better spacing between sections
   - ✅ Components stand alone without external margins
   - ✅ Improved alignment matching old template style

The modular system now provides a cleaner, more aligned appearance while maintaining all benefits of the component architecture.

## Article Layout Refinement (Completed 2025-07-06)

### Summary
Further refined the article layout to ensure proper alignment with navbar and improved callout styling:

1. **Alignment Fixes**:
   - Set article container to max-width 1400px (matching .container from base.html)
   - Added navbar-inner styling with same width and padding
   - Aligned hero content with article container
   - Text now properly aligns with header logo

2. **Callout Improvements**:
   - Reduced callout indentation to 80px for better visual hierarchy
   - Constrained callout width to maintain readability
   - Improved responsive behavior

3. **Full-width Sections**:
   - Fixed section-callout and section-wide to be truly full viewport width
   - Used proper CSS technique (position: relative; left: 50%; margin-left: -50vw)
   - Content within full-width sections still respects container constraints

The article layout now has consistent alignment throughout, with text starting at the same position as the header logo, properly indented callout boxes, and true full-width callout sections.

## Layout System Implementation (Completed 2025-07-06)

### Summary
Implemented the proper layout system from the old template to ensure consistent spacing and alignment:

1. **Container System**:
   - Standard container: 1500px (for navbar, hero, footer alignment)
   - Article container: 850px max-width for readable content
   - Wide container: 1800px for large tables/dashboards
   - Consistent padding: 48px desktop, 24px mobile

2. **Article Layout**:
   - Grid-based layout with sidebar support (1fr 240px)
   - Article content constrained to 850px for readability
   - Proper spacing using CSS variables
   - Single column on screens < 1200px

3. **Section Types**:
   - Full-width sections using `calc(-50vw + 50%)` technique
   - Callout boxes with 120px left indentation
   - Section content respects container constraints
   - Proper gradient backgrounds for callout sections

4. **Responsive Design**:
   - Breakpoints at 1200px (sidebar removal) and 768px (mobile)
   - Adjusted padding and font sizes for mobile
   - Maintained alignment across all screen sizes

The layout now properly matches the old template structure while maintaining the modular component system.

## Component Spacing and Presentation (Completed 2025-07-06)

### Summary
Added proper spacing between components in article sections:

1. **Component Spacing**:
   - Added `article-section-spacing` wrapper div around components
   - Provides consistent `var(--spacing-component)` margin between elements
   - Last child has no bottom margin to prevent extra space

2. **Template Changes**:
   - Wrapped callout_inline sections in spacing div
   - Wrapped article_content sections in spacing div
   - Wrapped initial article content in spacing div
   - Components themselves remain unchanged

The components now have proper spacing between them while maintaining the callout box indentation.

## Banner and Hero Header Spacing Fix (Completed 2025-07-06)

### Summary
Fixed spacing issues between banner and hero header, updated component types in manifest:

1. **Spacing Fixes**:
   - Removed extra padding-top from page-content (set to 0)
   - Hero header now starts immediately after navbar
   - No extra space between banner and hero

2. **Manifest Updates**:
   - Changed worst_performers from "raw" to "section_callout_dynamic"
   - Changed conclusion_text to "article_content" with parent_type: "article"
   - Changed full_results_intro to "article_content" with parent_type: "article"
   - Changed correlation_transition to "article_content" with parent_type: "article"

3. **Hero Header Styling**:
   - Added comprehensive hero header styles to article_styles.css
   - Proper gradient overlay with slight transparency (0.9 opacity)
   - Purple theme gradient for knowledge calibration pages
   - Hero content properly positioned above animation and gradient
   - Title, author, and subtitle styling matching reference

The page now properly aligns with the reference screenshot, with no extra spacing and correct component types throughout.

## Full-Width Callout Alignment Fix (Completed 2025-07-06)

### Summary
Fixed full-width callout sections alignment and component viewer display:

1. **Alignment Fixes**:
   - Updated section-callout CSS to use proper full-width technique
   - Changed from `calc(-50vw + 50%)` to `position: relative; left: 50%; margin-left: -50vw`
   - Applied same fix to section-wide and section-full classes
   - Ensures proper centering and full viewport width

2. **Dynamic Section Wrapper**:
   - Added proper wrapper for section_callout_dynamic in research_article.html
   - Now wraps dynamic content with section-callout structure
   - Includes title and text headers when specified in manifest

3. **Component Viewer Fix**:
   - Updated viewer to properly display section_callout_dynamic components
   - Added full wrapper structure with boundaries visualization
   - Now shows as "Full-Width Dynamic Section Callout" with proper styling

Full-width callout sections now properly extend to viewport edges and display correctly in both the main page and component viewer.

## Component Viewer and Gradient Visibility Fix (Completed 2025-07-06)

### Summary
Fixed component viewer display issues and improved gradient visibility:

1. **Component Viewer Fixes**:
   - Added minimum heights for flow diagram components
   - Ensured flow-diagram-wrapper has proper dimensions (min-height: 200px)
   - Set flow-diagram to center content and have minimum height
   - Added min-width and min-height to flow boxes for visibility
   - Fixed section-callout positioning within component viewer

2. **Gradient Visibility**:
   - Increased purple gradient opacity from 0.06-0.08 to 0.10-0.12
   - Increased border opacity from 0.2 to 0.3 for better definition
   - Makes the full-width sections more visually distinct

3. **Full-Width Behavior**:
   - The gradient background correctly extends to viewport edges
   - Content is intentionally constrained to 1500px for readability
   - This is the expected behavior for optimal user experience

The component viewer now properly displays flow diagrams and other full-width sections, and the gradient backgrounds are more visible while maintaining design consistency.

## Section Callout Dynamic Rendering Fix (Completed 2025-07-06)

### Summary
Fixed the horizontal scrolling issue in the component viewer when displaying section_callout_dynamic sections. The content was appearing shifted to the right, requiring users to scroll horizontally to see it properly.

### Changes Made

1. **Updated component viewer.html** (lines 471-488):
   - Modified the section_callout_dynamic rendering to use the section_callout_wrapper.html partial
   - This ensures consistency with how other section callouts are rendered

2. **Updated section_callout_wrapper.html**:
   - Added conditional logic to use different CSS classes based on context
   - When `in_viewer` is true, uses `section-callout-component` classes
   - When `in_viewer` is false, uses `section-callout` classes for full-width behavior

3. **Added CSS overrides in viewer.html** (lines 351-387):
   - Added styles to prevent horizontal scrolling in the component viewer
   - Overrode full-width margins for section-callout-component
   - Added overflow-x: hidden to prevent horizontal scrolling
   - Ensured proper content containment with max-width constraints

### Result
The section_callout_dynamic sections now display correctly in the component viewer without horizontal scrolling, while maintaining their full-width behavior in the actual article pages.

## Section Width and Layout Issues Fix (Completed 2025-07-06)

### Summary
Fixed section width issues where full-width sections weren't properly extending to viewport edges. The issue was caused by missing positioning properties and potential conflicts with parent container overflow settings.

### Changes Made

1. **Enhanced Full-Width CSS in article_styles.css**:
   - Added `position: relative` and `left: 0` to ensure proper alignment
   - Added `transform: translateX(0)` to ensure sections break out properly even with body overflow-x: hidden
   - Applied fixes to `.section-callout`, `.section-wide`, and `.section-full` classes
   - Added documentation explaining the full-width technique

2. **Updated Parent Container Overflow**:
   - Set `overflow-x: visible` on `.main-content` and `.page-content` to allow sections to break out
   - Added `position: relative` to body element for proper positioning context

3. **Added Responsive Adjustments**:
   - Ensured full-width sections work properly on mobile devices
   - Added explicit width and margin calculations for mobile breakpoints

4. **Documentation**:
   - Added comprehensive comment block explaining the full-width technique
   - Documented that width: 100vw makes sections viewport width
   - Explained that margin-left: calc(-50vw + 50%) breaks out of parent containers
   - Noted that parent containers must have overflow-x: visible for proper functionality

### Result
Full-width sections now properly extend to viewport edges on all screen sizes while maintaining proper content alignment. The technique is robust and works even when body has overflow-x: hidden.

## Component Viewer Dynamic Section Fixes (Completed 2025-07-06)

### Summary
Fixed issues with dynamic full-width callout sections not displaying properly in the component viewer. The main problem was timing issues with component initialization and missing component viewer detection in some components.

### Changes Made

1. **Fixed 26_top_models_web_alignment.html**:
   - Added component viewer detection to prevent auto-initialization conflicts
   - Added console logging with [TopModelsWebAlignment] prefix for debugging
   - Exposed `initTopModelsWebAlignment` function for manual initialization
   - Updated to follow same pattern as other dynamic components

2. **Enhanced Component Viewer (viewer.html)**:
   - Added robust initialization sequence with retry logic
   - Implemented `initializeComponent` function with exponential backoff
   - Added debugging information to check content area existence
   - Updated initialization calls for all dynamic components
   - Added proper error handling and console logging

3. **Created Component Patterns Documentation**:
   - Created COMPONENT_PATTERNS.md with standard patterns for components
   - Documented requirements for component viewer detection
   - Provided template code for creating new dynamic components
   - Added troubleshooting guide for common issues

### Technical Details
- Dynamic components must check for `/components/` in URL path
- Components should expose global initialization functions
- Component viewer uses retry logic (5 attempts with exponential backoff)
- All components use consistent logging with [ComponentName] prefix

### Result
All dynamic full-width callout sections now properly display in the component viewer with data loading correctly from API endpoints. The system is more robust with better error handling and debugging capabilities.
