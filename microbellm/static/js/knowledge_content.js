// Knowledge Calibration page dynamic content loader
// This file handles loading dynamic content for the knowledge calibration sections

// Load top performers section
function loadTopPerformers() {
    const container = document.getElementById('topPerformersContainer');
    if (!container) return;
    
    // The actual rendering is handled by the main loadKnowledgeAnalysisData function
    if (window.knowledgeData) {
        const { modelData, templateArray } = window.knowledgeData;
        const overallModelScores = calculateOverallModelScores(modelData, templateArray);
        const sortedModels = Object.entries(overallModelScores)
            .map(([modelName, scores]) => ({ modelName, ...scores }))
            .sort((a, b) => b.averageQualityScore - a.averageQualityScore);
        
        renderTopPerformers(sortedModels);
    }
}

// Load dynamic stats
function loadDynamicStats() {
    if (window.knowledgeData) {
        const { modelData, templateArray } = window.knowledgeData;
        updateDynamicStats(modelData, templateArray);
    }
}

// Load score calculation example
function loadScoreCalculationExample() {
    if (window.knowledgeData) {
        const { modelData, templateArray, inputTypeArray } = window.knowledgeData;
        renderScoreExample(modelData, templateArray, inputTypeArray);
    }
}

// Make functions available globally
window.loadTopPerformers = loadTopPerformers;
window.loadDynamicStats = loadDynamicStats;
window.loadScoreCalculationExample = loadScoreCalculationExample;