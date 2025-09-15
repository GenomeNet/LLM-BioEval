/**
 * SVG Export Utility for Components
 * Converts HTML components to downloadable SVG files
 */

class SVGExporter {
    constructor() {
        this.xmlns = "http://www.w3.org/2000/svg";
        this.xlinkns = "http://www.w3.org/1999/xlink";
    }

    /**
     * Detect component type
     */
    detectComponentType(element) {
        if (element.querySelector('.phenotype-card')) {
            return 'phenotype-cards';
        }
        if (element.querySelector('.section-callout')) {
            return 'section-callout';
        }
        if (element.querySelector('.callout')) {
            return 'callout-inline';
        }
        if (element.querySelector('.accuracy-table-wrapper')) {
            return 'data-table';
        }
        if (element.querySelector('canvas')) {
            return 'canvas-visualization';
        }
        return 'generic';
    }

    /**
     * Export an HTML element as SVG
     * @param {HTMLElement} element - The element to export
     * @param {Object} options - Export options
     * @returns {Promise<string>} - SVG string
     */
    async exportToSVG(element, options = {}) {
        const defaults = {
            width: element.offsetWidth || 1200,
            height: element.offsetHeight || 800,
            backgroundColor: '#ffffff',
            padding: 40,
            includeStyles: true,
            fontFamily: 'Arial, Helvetica, sans-serif'
        };
        
        const config = { ...defaults, ...options };
        
        // Adjust dimensions based on component type
        const componentType = this.detectComponentType(element);
        if (componentType === 'section-callout' || componentType === 'callout-inline') {
            // Add extra padding for callout components
            config.padding = 60;
        }
        
        // Create SVG element
        const svg = document.createElementNS(this.xmlns, 'svg');
        svg.setAttribute('width', config.width + config.padding * 2);
        svg.setAttribute('height', config.height + config.padding * 2);
        svg.setAttribute('viewBox', `0 0 ${config.width + config.padding * 2} ${config.height + config.padding * 2}`);
        svg.setAttribute('xmlns', this.xmlns);
        svg.setAttribute('xmlns:xlink', this.xlinkns);
        
        // Add background
        const bg = document.createElementNS(this.xmlns, 'rect');
        bg.setAttribute('width', '100%');
        bg.setAttribute('height', '100%');
        bg.setAttribute('fill', config.backgroundColor);
        svg.appendChild(bg);
        
        // Create foreign object to embed HTML
        const foreignObject = document.createElementNS(this.xmlns, 'foreignObject');
        foreignObject.setAttribute('x', config.padding);
        foreignObject.setAttribute('y', config.padding);
        foreignObject.setAttribute('width', config.width);
        foreignObject.setAttribute('height', config.height);
        
        // Clone the element
        const clonedElement = element.cloneNode(true);
        
        // Create a container div with styles
        const container = document.createElement('div');
        container.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
        container.style.fontFamily = config.fontFamily;
        container.style.fontSize = '14px';
        container.style.lineHeight = '1.6';
        container.style.color = '#1f2937';
        
        if (config.includeStyles) {
            // Embed critical styles
            const style = document.createElement('style');
            style.textContent = this.getCriticalCSS(element);
            container.appendChild(style);
        }
        
        container.appendChild(clonedElement);
        foreignObject.appendChild(container);
        svg.appendChild(foreignObject);
        
        return new XMLSerializer().serializeToString(svg);
    }

    /**
     * Export as pure SVG graphics (for phenotype cards)
     * @param {HTMLElement} element - The element to export
     * @param {Object} options - Export options
     * @returns {string} - SVG string
     */
    exportPhenotypesAsSVG(element, options = {}) {
        const cards = element.querySelectorAll('.phenotype-card');
        const config = {
            cardWidth: 340,
            cardHeight: 180,
            padding: 20,
            columns: 3,
            gap: 20,
            ...options
        };
        
        const rows = Math.ceil(cards.length / config.columns);
        const svgWidth = config.columns * config.cardWidth + (config.columns - 1) * config.gap + config.padding * 2;
        const svgHeight = rows * config.cardHeight + (rows - 1) * config.gap + config.padding * 2;
        
        // Create SVG
        const svg = document.createElementNS(this.xmlns, 'svg');
        svg.setAttribute('width', svgWidth);
        svg.setAttribute('height', svgHeight);
        svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
        svg.setAttribute('xmlns', this.xmlns);
        
        // Add definitions for gradients
        const defs = document.createElementNS(this.xmlns, 'defs');
        
        // Green gradient
        const greenGradient = document.createElementNS(this.xmlns, 'linearGradient');
        greenGradient.setAttribute('id', 'greenGradient');
        greenGradient.setAttribute('x1', '0%');
        greenGradient.setAttribute('y1', '0%');
        greenGradient.setAttribute('x2', '100%');
        greenGradient.setAttribute('y2', '100%');
        
        const stop1 = document.createElementNS(this.xmlns, 'stop');
        stop1.setAttribute('offset', '0%');
        stop1.setAttribute('stop-color', '#4ade80');
        stop1.setAttribute('stop-opacity', '0.1');
        greenGradient.appendChild(stop1);
        
        const stop2 = document.createElementNS(this.xmlns, 'stop');
        stop2.setAttribute('offset', '100%');
        stop2.setAttribute('stop-color', '#22c55e');
        stop2.setAttribute('stop-opacity', '0.1');
        greenGradient.appendChild(stop2);
        
        defs.appendChild(greenGradient);
        svg.appendChild(defs);
        
        // Background
        const bg = document.createElementNS(this.xmlns, 'rect');
        bg.setAttribute('width', '100%');
        bg.setAttribute('height', '100%');
        bg.setAttribute('fill', '#ffffff');
        svg.appendChild(bg);
        
        // Create cards
        cards.forEach((card, index) => {
            const col = index % config.columns;
            const row = Math.floor(index / config.columns);
            const x = config.padding + col * (config.cardWidth + config.gap);
            const y = config.padding + row * (config.cardHeight + config.gap);
            
            // Card group
            const g = document.createElementNS(this.xmlns, 'g');
            g.setAttribute('transform', `translate(${x}, ${y})`);
            
            // Card background
            const rect = document.createElementNS(this.xmlns, 'rect');
            rect.setAttribute('width', config.cardWidth);
            rect.setAttribute('height', config.cardHeight);
            rect.setAttribute('rx', '12');
            rect.setAttribute('fill', '#ffffff');
            rect.setAttribute('stroke', '#e5e7eb');
            rect.setAttribute('stroke-width', '2');
            g.appendChild(rect);
            
            // Card content
            const title = card.querySelector('.phenotype-title')?.textContent || '';
            const type = card.querySelector('.phenotype-type')?.textContent || '';
            const description = card.querySelector('.phenotype-description')?.textContent || '';
            const values = Array.from(card.querySelectorAll('.value-tag')).map(v => v.textContent);
            
            // Title
            const titleText = document.createElementNS(this.xmlns, 'text');
            titleText.setAttribute('x', '20');
            titleText.setAttribute('y', '35');
            titleText.setAttribute('font-family', 'Arial, Helvetica, sans-serif');
            titleText.setAttribute('font-size', '18');
            titleText.setAttribute('font-weight', '700');
            titleText.setAttribute('fill', '#111827');
            titleText.textContent = title;
            g.appendChild(titleText);
            
            // Type badge
            const typeBg = document.createElementNS(this.xmlns, 'rect');
            typeBg.setAttribute('x', config.cardWidth - 100);
            typeBg.setAttribute('y', '20');
            typeBg.setAttribute('width', '80');
            typeBg.setAttribute('height', '24');
            typeBg.setAttribute('rx', '4');
            typeBg.setAttribute('fill', 'url(#greenGradient)');
            g.appendChild(typeBg);
            
            const typeText = document.createElementNS(this.xmlns, 'text');
            typeText.setAttribute('x', config.cardWidth - 60);
            typeText.setAttribute('y', '36');
            typeText.setAttribute('font-family', 'Arial, Helvetica, sans-serif');
            typeText.setAttribute('font-size', '11');
            typeText.setAttribute('font-weight', '600');
            typeText.setAttribute('text-anchor', 'middle');
            typeText.setAttribute('fill', '#059669');
            typeText.textContent = type.toUpperCase();
            g.appendChild(typeText);
            
            // Description (wrap text)
            const descLines = this.wrapText(description, 50);
            descLines.forEach((line, i) => {
                const descText = document.createElementNS(this.xmlns, 'text');
                descText.setAttribute('x', '20');
                descText.setAttribute('y', 60 + i * 18);
                descText.setAttribute('font-family', 'Arial, Helvetica, sans-serif');
                descText.setAttribute('font-size', '14');
                descText.setAttribute('fill', '#6b7280');
                descText.textContent = line;
                g.appendChild(descText);
            });
            
            // Value tags
            let tagX = 20;
            const tagY = 130;
            values.forEach((value) => {
                const tagWidth = value.length * 7 + 24;
                
                // Tag background
                const tagBg = document.createElementNS(this.xmlns, 'rect');
                tagBg.setAttribute('x', tagX);
                tagBg.setAttribute('y', tagY);
                tagBg.setAttribute('width', tagWidth);
                tagBg.setAttribute('height', '26');
                tagBg.setAttribute('rx', '6');
                tagBg.setAttribute('fill', 'url(#greenGradient)');
                tagBg.setAttribute('stroke', '#86efac');
                tagBg.setAttribute('stroke-width', '1');
                g.appendChild(tagBg);
                
                // Tag text
                const tagText = document.createElementNS(this.xmlns, 'text');
                tagText.setAttribute('x', tagX + tagWidth / 2);
                tagText.setAttribute('y', tagY + 17);
                tagText.setAttribute('font-family', 'Arial, Helvetica, sans-serif');
                tagText.setAttribute('font-size', '13');
                tagText.setAttribute('font-weight', '500');
                tagText.setAttribute('text-anchor', 'middle');
                tagText.setAttribute('fill', '#047857');
                tagText.textContent = value;
                g.appendChild(tagText);
                
                tagX += tagWidth + 8;
            });
            
            svg.appendChild(g);
        });
        
        return new XMLSerializer().serializeToString(svg);
    }

    /**
     * Wrap text to fit within a certain character limit
     */
    wrapText(text, maxChars) {
        const words = text.split(' ');
        const lines = [];
        let currentLine = '';
        
        words.forEach(word => {
            if ((currentLine + word).length <= maxChars) {
                currentLine += (currentLine ? ' ' : '') + word;
            } else {
                if (currentLine) lines.push(currentLine);
                currentLine = word;
            }
        });
        
        if (currentLine) lines.push(currentLine);
        return lines.slice(0, 2); // Max 2 lines
    }

    /**
     * Get critical CSS for the element
     */
    getCriticalCSS(element) {
        return `
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            .phenotype-cards-container {
                width: 100%;
                padding: 20px;
            }
            
            .phenotype-cards {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
            }
            
            .phenotype-card {
                background: #ffffff;
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                padding: 24px;
            }
            
            .phenotype-card-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 12px;
            }
            
            .phenotype-title {
                font-size: 18px;
                font-weight: 700;
                color: #111827;
                margin: 0;
            }
            
            .phenotype-type {
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: #059669;
                background: rgba(74, 222, 128, 0.1);
                padding: 4px 8px;
                border-radius: 4px;
            }
            
            .phenotype-description {
                font-size: 14px;
                color: #6b7280;
                line-height: 1.6;
                margin-bottom: 16px;
            }
            
            .phenotype-values {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            
            .value-tag {
                display: inline-block;
                padding: 6px 12px;
                background: rgba(74, 222, 128, 0.08);
                border: 1px solid rgba(34, 197, 94, 0.2);
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                color: #047857;
            }
        `;
    }

    /**
     * Export callout components as clean SVG
     */
    exportCalloutAsSVG(element, options = {}) {
        const config = {
            padding: 40,
            backgroundColor: '#ffffff',
            includeGradient: true,
            fontFamily: 'Arial, Helvetica, sans-serif',
            ...options
        };
        
        // Get actual dimensions
        const rect = element.getBoundingClientRect();
        const width = Math.max(rect.width, 800);
        const height = Math.max(rect.height, 400);
        
        const svg = document.createElementNS(this.xmlns, 'svg');
        svg.setAttribute('width', width + config.padding * 2);
        svg.setAttribute('height', height + config.padding * 2);
        svg.setAttribute('viewBox', `0 0 ${width + config.padding * 2} ${height + config.padding * 2}`);
        svg.setAttribute('xmlns', this.xmlns);
        
        // Add definitions
        const defs = document.createElementNS(this.xmlns, 'defs');
        
        if (config.includeGradient) {
            // Gradient for callout backgrounds
            const gradient = document.createElementNS(this.xmlns, 'linearGradient');
            gradient.setAttribute('id', 'calloutGradient');
            gradient.setAttribute('x1', '0%');
            gradient.setAttribute('y1', '0%');
            gradient.setAttribute('x2', '100%');
            gradient.setAttribute('y2', '100%');
            
            const stop1 = document.createElementNS(this.xmlns, 'stop');
            stop1.setAttribute('offset', '0%');
            stop1.setAttribute('stop-color', element.classList.contains('section-callout--green') ? '#4ade80' : '#a78bfa');
            stop1.setAttribute('stop-opacity', '0.05');
            gradient.appendChild(stop1);
            
            const stop2 = document.createElementNS(this.xmlns, 'stop');
            stop2.setAttribute('offset', '100%');
            stop2.setAttribute('stop-color', element.classList.contains('section-callout--green') ? '#22c55e' : '#7c3aed');
            stop2.setAttribute('stop-opacity', '0.1');
            gradient.appendChild(stop2);
            
            defs.appendChild(gradient);
        }
        
        svg.appendChild(defs);
        
        // Background
        const bg = document.createElementNS(this.xmlns, 'rect');
        bg.setAttribute('width', '100%');
        bg.setAttribute('height', '100%');
        bg.setAttribute('fill', config.backgroundColor);
        svg.appendChild(bg);
        
        // Callout container
        const calloutBg = document.createElementNS(this.xmlns, 'rect');
        calloutBg.setAttribute('x', config.padding);
        calloutBg.setAttribute('y', config.padding);
        calloutBg.setAttribute('width', width);
        calloutBg.setAttribute('height', height);
        calloutBg.setAttribute('rx', '12');
        calloutBg.setAttribute('fill', config.includeGradient ? 'url(#calloutGradient)' : '#f9fafb');
        calloutBg.setAttribute('stroke', '#e5e7eb');
        calloutBg.setAttribute('stroke-width', '1');
        svg.appendChild(calloutBg);
        
        // Foreign object for HTML content
        const foreignObject = document.createElementNS(this.xmlns, 'foreignObject');
        foreignObject.setAttribute('x', config.padding + 20);
        foreignObject.setAttribute('y', config.padding + 20);
        foreignObject.setAttribute('width', width - 40);
        foreignObject.setAttribute('height', height - 40);
        
        const container = document.createElement('div');
        container.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
        container.style.fontFamily = config.fontFamily;
        container.style.fontSize = '14px';
        container.style.lineHeight = '1.6';
        container.style.color = '#1f2937';
        
        // Clone and clean the content
        const clonedElement = element.cloneNode(true);
        // Remove any script tags
        clonedElement.querySelectorAll('script').forEach(script => script.remove());
        // Remove loading states
        clonedElement.querySelectorAll('.loading-overlay').forEach(el => el.remove());
        
        container.appendChild(clonedElement);
        foreignObject.appendChild(container);
        svg.appendChild(foreignObject);
        
        return new XMLSerializer().serializeToString(svg);
    }

    /**
     * Export data tables as SVG
     */
    exportTableAsSVG(element, options = {}) {
        const table = element.querySelector('table') || element;
        if (!table || table.tagName !== 'TABLE') {
            return this.exportToSVG(element, options);
        }
        
        const config = {
            cellPadding: 10,
            fontSize: 12,
            headerColor: '#f3f4f6',
            borderColor: '#e5e7eb',
            fontFamily: 'Arial, Helvetica, sans-serif',
            ...options
        };
        
        // Calculate dimensions
        const rows = table.querySelectorAll('tr');
        const cols = Math.max(...Array.from(rows).map(row => row.querySelectorAll('td, th').length));
        const cellWidth = 150;
        const cellHeight = 40;
        const svgWidth = cols * cellWidth + 40;
        const svgHeight = rows.length * cellHeight + 40;
        
        const svg = document.createElementNS(this.xmlns, 'svg');
        svg.setAttribute('width', svgWidth);
        svg.setAttribute('height', svgHeight);
        svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
        svg.setAttribute('xmlns', this.xmlns);
        
        // Background
        const bg = document.createElementNS(this.xmlns, 'rect');
        bg.setAttribute('width', '100%');
        bg.setAttribute('height', '100%');
        bg.setAttribute('fill', '#ffffff');
        svg.appendChild(bg);
        
        // Draw table
        let y = 20;
        rows.forEach((row, rowIndex) => {
            const cells = row.querySelectorAll('td, th');
            let x = 20;
            
            cells.forEach((cell, colIndex) => {
                // Cell background
                const cellBg = document.createElementNS(this.xmlns, 'rect');
                cellBg.setAttribute('x', x);
                cellBg.setAttribute('y', y);
                cellBg.setAttribute('width', cellWidth);
                cellBg.setAttribute('height', cellHeight);
                cellBg.setAttribute('fill', cell.tagName === 'TH' ? config.headerColor : '#ffffff');
                cellBg.setAttribute('stroke', config.borderColor);
                cellBg.setAttribute('stroke-width', '1');
                svg.appendChild(cellBg);
                
                // Cell text
                const text = document.createElementNS(this.xmlns, 'text');
                text.setAttribute('x', x + config.cellPadding);
                text.setAttribute('y', y + cellHeight / 2 + 4);
                text.setAttribute('font-family', config.fontFamily);
                text.setAttribute('font-size', config.fontSize);
                text.setAttribute('font-weight', cell.tagName === 'TH' ? 'bold' : 'normal');
                text.setAttribute('fill', '#374151');
                
                // Truncate long text
                let textContent = cell.textContent.trim();
                if (textContent.length > 20) {
                    textContent = textContent.substring(0, 17) + '...';
                }
                text.textContent = textContent;
                svg.appendChild(text);
                
                x += cellWidth;
            });
            
            y += cellHeight;
        });
        
        return new XMLSerializer().serializeToString(svg);
    }

    /**
     * Download SVG as file
     */
    downloadSVG(svgString, filename = 'component-export.svg') {
        const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

// Make globally available
window.SVGExporter = SVGExporter;