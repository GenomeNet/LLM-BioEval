# Project plan 

## Project overview 

MicrobeLLM is a Python tool designed to evaluate Large Language Models (LLMs) on their ability to predict microbial phenotypes. This tool helps researchers assess how well different LLMs perform on microbiological tasks by querying them with bacterial species names and comparing their predictions against known characteristics.

## TODO

- [ ] make the 4 boxes in "200 total transformations" slightly smaller that we can display two of these side by side e.g. that we have like a grid of 2x2 boxes there that we save a bit vertical space
- [ ] the vertiacl space after "With our 200 fabricated names in hand, we run a fully automated loop: each name is inserted into every query template, each template is sent to every model exposed through the OpenRouter API, and every reply is stored with a timestamp and model ID. A small validation script then checks whether the response is one of the allowed labels (limited, intermediate, extensive, or NA) and scores it against the ground truth (which, for fictional species, should be NA or limited)." and the first "Full-Width Callout Sections" `.section-callout` it to big, can you remove half of the space there 
- [ ] the vertical space after "Top Performing Models" full width callout and the text content afterwords starting with "However, there is a significant difference in how models perform. The models with the most room for improvement are:" is too big, correct that, try to update also the Style guide that this is more consistent and smaller in the future 
- [ ] the first few tables (one per template) that we have e.g. "template1_knowlege" should be more like a Full-Width Callout Sections since they are quite wide 
- [ ] get rid of the boarder around the barplot at "This chart shows quality scores for each model stratified by template. Higher scores indicate better performance at recognizing fictional species. Only models with results for all templates are shown. and update the style guide that we reflect these in the future 
- [ ] the horizontal width of the barplot is too big, try to remove the space between the bars further 
- [ ] the first animation in Web-Aligned Knowledge: Real Bacterial Names vs. Google Counts should be a Full-Width Callout Sections, currenlty its an intented callout box. 
- [ ] text afterwoards sould be Standard Article Sections, currently its going over the full page which is against style guide. 
- [ ] table "Correlation Score Distribution" should be a Full-Width Callout Section
- [ ] there should be normal Standard Article Sections after the Correlation Score Distribution" table, think about text and add it there, will change it later, maybe this can introduce the "Top Models – Web-Alignment" section that is then afterwards
- [ ] the plot under "Correlation Visualization: Google Search vs. Knowledge Level" should be a "Full-Width Callout Sections" also the legend should follow style guide add text (Standard Article Sections) before and after that I will change later
