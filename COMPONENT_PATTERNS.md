# Component Patterns Guide

This guide documents the standard patterns for creating components in the microbeLLM modular research page system.

## Component Types

### 1. Text Sections (`article`)
- Simple HTML/Markdown content for describing data
- Rendered with standard article styling
- Maximum width constrained for readability

### 2. Inline Callouts (`callout_inline`)
- Highlighted content boxes within article sections
- Right-indented with colored backgrounds
- Used for definitions, examples, and asides

### 3. Full-Width Callouts (`section_callout` and `section_callout_dynamic`)
- Full viewport width sections with gradient backgrounds
- Used for major results, tables, and visualizations
- Dynamic variants load data via API calls

## Creating Dynamic Components

### Pattern for Self-Contained Dynamic Components

```javascript
// Self-contained script for [Component Name]
(function() {
    console.log('[ComponentName] Script starting...');
    
    // Wait a bit for DOM to settle
    setTimeout(function() {
        console.log('[ComponentName] Looking for container...');
        // Do not auto-init in component viewer context
        if (!window.location.pathname.includes('/components/')) {
            initializeComponent();
        }
    }, 100);
    
    // Main initialization function
    function initializeComponent() {
        console.log('[ComponentName] Initializing...');
        loadComponentData();
    }
    
    // Expose initialization function for component viewer
    window.initComponentName = function() {
        console.log('[ComponentName] Manual initialization triggered');
        loadComponentData();
    };
    
    // Data loading function
    function loadComponentData() {
        console.log('[ComponentName] Loading data...');
        const container = document.getElementById('componentContainer');
        if (!container) {
            console.error('[ComponentName] Container not found!');
            return;
        }
        
        // Fetch data from API
        fetch('/api/endpoint')
            .then(response => response.json())
            .then(data => {
                console.log('[ComponentName] Data received:', data);
                renderComponent(data);
            })
            .catch(error => {
                console.error('[ComponentName] Error loading data:', error);
                container.innerHTML = '<div class="error-message">Failed to load data</div>';
            });
    }
    
    // Rendering function
    function renderComponent(data) {
        // Render your component here
    }
})();
```

### Key Requirements

1. **Component Viewer Detection**: Check for `/components/` in the URL path to prevent auto-initialization
2. **Manual Initialization**: Expose a global function for the component viewer to call
3. **Consistent Logging**: Use `[ComponentName]` prefix for all console logs
4. **Error Handling**: Always handle missing containers and API errors gracefully
5. **Loading States**: Show loading indicators while fetching data

## Component Viewer Integration

The component viewer (`viewer.html`) handles initialization differently based on component type:

### For Components with Manual Init
```javascript
{% if section.id == 'your_component' %}
    initializeComponent('your_component', 'initYourComponent');
{% endif %}
```

### For Self-Initializing Components
```javascript
{% if section.id == 'auto_component' %}
    console.log('Viewer: auto_component will auto-initialize');
{% endif %}
```

## Manifest Configuration

In your `manifest.yaml`:

```yaml
sections:
  - id: "component_id"
    type: "section_callout_dynamic"
    file: "sections/component_file.html"
    title: "Component Title"
    text: "Component description"
    container_id: "optionalContainerId"
```

## API Integration

Dynamic components typically fetch data from API endpoints:

- `/api/knowledge_analysis_data` - Knowledge analysis data
- `/api/search_count_correlation` - Correlation data
- Other custom endpoints as needed

## CSS Classes

### For Section Callouts
- `.section-callout` - Base class for production pages
- `.section-callout-component` - Modified class for component viewer
- `.section-callout--purple` - Purple theme variant
- `.section-callout--green` - Green theme variant

### Content Containers
- `.section-callout__content` - Standard width content
- `.section-callout__content--full-width` - Full width content

## Troubleshooting

### Component Not Showing in Viewer
1. Check browser console for initialization logs
2. Verify the component ID matches in manifest and viewer.html
3. Ensure API endpoints are accessible
4. Check for CSS conflicts hiding content

### Initialization Issues
1. Add more console logging to track execution
2. Increase retry attempts in viewer's initialization
3. Check for JavaScript errors in component code
4. Verify global functions are properly exposed

### API Data Issues
1. Check network tab for API responses
2. Verify API endpoint paths are correct
3. Check for CORS issues
4. Ensure proper error handling in fetch calls