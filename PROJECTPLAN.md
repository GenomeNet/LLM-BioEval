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