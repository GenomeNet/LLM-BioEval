# Best Model-Phenotype Combinations (Balanced Accuracy)

Generated from MicrobeLLM API data using balanced accuracy methodology

*Note: Excludes aerophilicity, health_association, and hemolysis phenotypes*


| Phenotype | Best Model | Model (Short) | Balanced Accuracy | Sample Size |
|-----------|------------|---------------|-------------------|-------------|
| Spore Formation | google/gemini-2.5-pro | gemini-2.5-pro | 96.6% | 3,612 |
| Cell Shape | openai/gpt-4.1-nano | gpt-4.1-nano | 91.7% | 3,290 |
| Biosafety Level | anthropic/claude-3.5-sonnet | claude-3.5-sonnet | 91.4% | 3,708 |
| Motility | openai/gpt-5 | gpt-5 | 90.9% | 2,949 |
| Animal Pathogenicity | google/gemini-flash-1.5 | gemini-flash-1.5 | 84.9% | 2,050 |
| Host Association | openai/gpt-4o | gpt-4o | 80.8% | 2,665 |
| Plant Pathogenicity | google/gemini-pro-1.5 | gemini-pro-1.5 | 79.4% | 3,723 |
| Extreme Environment Tolerance | deepseek/deepseek-r1 | deepseek-r1 | 72.5% | 3,350 |
| Gram Staining | google/gemini-2.5-pro | gemini-2.5-pro | 69.5% | 3,840 |
| Biofilm Formation | openai/gpt-4 | gpt-4 | 61.7% | 253 |

## Summary Statistics

- Total phenotypes analyzed: 10
- Average best balanced accuracy: 81.9%

### Best Models Distribution (by short name):

- gemini-2.5-pro: 2 phenotypes
- gpt-5: 1 phenotypes
- deepseek-r1: 1 phenotypes
- gpt-4: 1 phenotypes
- gemini-flash-1.5: 1 phenotypes
- claude-3.5-sonnet: 1 phenotypes
- gpt-4o: 1 phenotypes
- gemini-pro-1.5: 1 phenotypes
- gpt-4.1-nano: 1 phenotypes