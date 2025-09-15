/**
 * Pure SVG Export - Creates true vector graphics without foreignObject
 * Compatible with Adobe Illustrator and other vector editors
 */

class PureSVGExporter {
    constructor() {
        this.xmlns = "http://www.w3.org/2000/svg";
        this.defaultFont = 'Arial, Helvetica, sans-serif';
    }

    /**
     * Export HTML table as pure SVG
     */
    exportTableAsPureSVG(element) {
        const table = element.querySelector('table');
        if (!table) return this.exportGenericAsPureSVG(element);

        // Get table dimensions
        const rows = Array.from(table.querySelectorAll('tr'));
        const maxCols = Math.max(...rows.map(row => row.querySelectorAll('td, th').length));
        
        const cellWidth = 180;
        const cellHeight = 40;
        const padding = 20;
        const fontSize = 12;
        
        const svgWidth = maxCols * cellWidth + padding * 2;
        const svgHeight = rows.length * cellHeight + padding * 2 + 60; // Extra space for title
        
        // Create SVG
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
        
        // Get title if present
        const titleElement = element.querySelector('.section-callout__title');
        if (titleElement) {
            const titleText = document.createElementNS(this.xmlns, 'text');
            titleText.setAttribute('x', svgWidth / 2);
            titleText.setAttribute('y', 30);
            titleText.setAttribute('font-family', this.defaultFont);
            titleText.setAttribute('font-size', '20');
            titleText.setAttribute('font-weight', 'bold');
            titleText.setAttribute('text-anchor', 'middle');
            titleText.setAttribute('fill', '#111827');
            titleText.textContent = titleElement.textContent;
            svg.appendChild(titleText);
        }
        
        // Draw table
        let yOffset = titleElement ? 70 : padding;
        
        rows.forEach((row, rowIndex) => {
            const cells = row.querySelectorAll('td, th');
            let xOffset = padding;
            
            cells.forEach((cell, colIndex) => {
                // Cell background
                const rect = document.createElementNS(this.xmlns, 'rect');
                rect.setAttribute('x', xOffset);
                rect.setAttribute('y', yOffset);
                rect.setAttribute('width', cellWidth);
                rect.setAttribute('height', cellHeight);
                rect.setAttribute('fill', cell.tagName === 'TH' ? '#f3f4f6' : '#ffffff');
                rect.setAttribute('stroke', '#e5e7eb');
                rect.setAttribute('stroke-width', '1');
                svg.appendChild(rect);
                
                // Cell text
                const text = document.createElementNS(this.xmlns, 'text');
                text.setAttribute('x', xOffset + 10);
                text.setAttribute('y', yOffset + cellHeight / 2 + 4);
                text.setAttribute('font-family', this.defaultFont);
                text.setAttribute('font-size', fontSize);
                text.setAttribute('font-weight', cell.tagName === 'TH' ? 'bold' : 'normal');
                text.setAttribute('fill', '#374151');
                
                // Handle bar charts in cells
                const barChart = cell.querySelector('.bar-chart-container');
                if (barChart) {
                    // Draw bar chart
                    const segments = barChart.querySelectorAll('.bar-segment');
                    let barX = xOffset + 10;
                    const barY = yOffset + 10;
                    const barHeight = 20;
                    const totalWidth = cellWidth - 20;
                    
                    segments.forEach(segment => {
                        const widthPercent = parseFloat(segment.style.width) / 100;
                        const segmentWidth = totalWidth * widthPercent;
                        
                        const barRect = document.createElementNS(this.xmlns, 'rect');
                        barRect.setAttribute('x', barX);
                        barRect.setAttribute('y', barY);
                        barRect.setAttribute('width', segmentWidth);
                        barRect.setAttribute('height', barHeight);
                        barRect.setAttribute('rx', '2');
                        
                        // Set color based on class
                        let fillColor = '#e5e7eb';
                        if (segment.classList.contains('na')) fillColor = '#e5e7eb';
                        else if (segment.classList.contains('limited')) fillColor = '#fbbf24';
                        else if (segment.classList.contains('moderate')) fillColor = '#60a5fa';
                        else if (segment.classList.contains('extensive')) fillColor = '#34d399';
                        
                        barRect.setAttribute('fill', fillColor);
                        svg.appendChild(barRect);
                        
                        // Add percentage text
                        if (segmentWidth > 30) {
                            const barText = document.createElementNS(this.xmlns, 'text');
                            barText.setAttribute('x', barX + segmentWidth / 2);
                            barText.setAttribute('y', barY + barHeight / 2 + 4);
                            barText.setAttribute('font-family', this.defaultFont);
                            barText.setAttribute('font-size', '10');
                            barText.setAttribute('text-anchor', 'middle');
                            barText.setAttribute('fill', '#ffffff');
                            barText.textContent = segment.textContent.trim();
                            svg.appendChild(barText);
                        }
                        
                        barX += segmentWidth;
                    });
                } else {
                    // Regular text
                    let cellText = cell.textContent.trim();
                    if (cellText.length > 25) {
                        cellText = cellText.substring(0, 22) + '...';
                    }
                    text.textContent = cellText;
                    svg.appendChild(text);
                }
                
                xOffset += cellWidth;
            });
            
            yOffset += cellHeight;
        });
        
        return new XMLSerializer().serializeToString(svg);
    }

    /**
     * Export callout sections as pure SVG
     */
    exportCalloutAsPureSVG(element) {
        // Get content dimensions
        const rect = element.getBoundingClientRect();
        const width = Math.min(rect.width || 1200, 1200);
        const height = Math.min(rect.height || 800, 800);
        const padding = 40;
        
        const svg = document.createElementNS(this.xmlns, 'svg');
        svg.setAttribute('width', width + padding * 2);
        svg.setAttribute('height', height + padding * 2);
        svg.setAttribute('viewBox', `0 0 ${width + padding * 2} ${height + padding * 2}`);
        svg.setAttribute('xmlns', this.xmlns);
        
        // Background
        const bg = document.createElementNS(this.xmlns, 'rect');
        bg.setAttribute('width', '100%');
        bg.setAttribute('height', '100%');
        bg.setAttribute('fill', '#ffffff');
        svg.appendChild(bg);
        
        // Callout background with gradient effect (simplified)
        const calloutBg = document.createElementNS(this.xmlns, 'rect');
        calloutBg.setAttribute('x', padding);
        calloutBg.setAttribute('y', padding);
        calloutBg.setAttribute('width', width);
        calloutBg.setAttribute('height', height);
        calloutBg.setAttribute('rx', '12');
        calloutBg.setAttribute('fill', element.classList.contains('section-callout--green') ? '#f0fdf4' : '#faf5ff');
        calloutBg.setAttribute('stroke', element.classList.contains('section-callout--green') ? '#bbf7d0' : '#e9d5ff');
        calloutBg.setAttribute('stroke-width', '1');
        svg.appendChild(calloutBg);
        
        // Extract text content
        let yOffset = padding + 40;
        
        // Title
        const title = element.querySelector('.section-callout__title, .callout-header h3');
        if (title) {
            const titleText = document.createElementNS(this.xmlns, 'text');
            titleText.setAttribute('x', padding + 30);
            titleText.setAttribute('y', yOffset);
            titleText.setAttribute('font-family', this.defaultFont);
            titleText.setAttribute('font-size', '24');
            titleText.setAttribute('font-weight', 'bold');
            titleText.setAttribute('fill', '#111827');
            titleText.textContent = title.textContent.trim();
            svg.appendChild(titleText);
            yOffset += 40;
        }
        
        // Description
        const description = element.querySelector('.section-callout__text, .callout-content p');
        if (description) {
            const lines = this.wrapText(description.textContent.trim(), 80);
            lines.forEach((line, i) => {
                const descText = document.createElementNS(this.xmlns, 'text');
                descText.setAttribute('x', padding + 30);
                descText.setAttribute('y', yOffset + i * 20);
                descText.setAttribute('font-family', this.defaultFont);
                descText.setAttribute('font-size', '14');
                descText.setAttribute('fill', '#6b7280');
                descText.textContent = line;
                svg.appendChild(descText);
            });
            yOffset += lines.length * 20 + 20;
        }
        
        // Handle any tables inside
        const table = element.querySelector('table');
        if (table) {
            const tableGroup = document.createElementNS(this.xmlns, 'g');
            tableGroup.setAttribute('transform', `translate(${padding + 20}, ${yOffset})`);
            
            // Add simplified table representation
            const rows = table.querySelectorAll('tr');
            let tableY = 0;
            
            rows.forEach((row, i) => {
                if (i > 5) return; // Limit rows for space
                
                const cells = row.querySelectorAll('td, th');
                let tableX = 0;
                
                cells.forEach((cell, j) => {
                    if (j > 3) return; // Limit columns
                    
                    const cellRect = document.createElementNS(this.xmlns, 'rect');
                    cellRect.setAttribute('x', tableX);
                    cellRect.setAttribute('y', tableY);
                    cellRect.setAttribute('width', '150');
                    cellRect.setAttribute('height', '30');
                    cellRect.setAttribute('fill', cell.tagName === 'TH' ? '#f3f4f6' : '#ffffff');
                    cellRect.setAttribute('stroke', '#e5e7eb');
                    cellRect.setAttribute('stroke-width', '1');
                    tableGroup.appendChild(cellRect);
                    
                    const cellText = document.createElementNS(this.xmlns, 'text');
                    cellText.setAttribute('x', tableX + 10);
                    cellText.setAttribute('y', tableY + 20);
                    cellText.setAttribute('font-family', this.defaultFont);
                    cellText.setAttribute('font-size', '12');
                    cellText.setAttribute('font-weight', cell.tagName === 'TH' ? 'bold' : 'normal');
                    cellText.setAttribute('fill', '#374151');
                    
                    let text = cell.textContent.trim();
                    if (text.length > 20) text = text.substring(0, 17) + '...';
                    cellText.textContent = text;
                    tableGroup.appendChild(cellText);
                    
                    tableX += 150;
                });
                
                tableY += 30;
            });
            
            svg.appendChild(tableGroup);
        }
        
        return new XMLSerializer().serializeToString(svg);
    }

    /**
     * Generic export for other components
     */
    exportGenericAsPureSVG(element) {
        const rect = element.getBoundingClientRect();
        const width = Math.min(rect.width || 800, 1200);
        const height = 400; // Fixed height for simplicity
        
        const svg = document.createElementNS(this.xmlns, 'svg');
        svg.setAttribute('width', width);
        svg.setAttribute('height', height);
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        svg.setAttribute('xmlns', this.xmlns);
        
        // Background
        const bg = document.createElementNS(this.xmlns, 'rect');
        bg.setAttribute('width', '100%');
        bg.setAttribute('height', '100%');
        bg.setAttribute('fill', '#ffffff');
        svg.appendChild(bg);
        
        // Extract and render text content
        let yOffset = 40;
        
        // Find all text elements
        const textElements = element.querySelectorAll('h1, h2, h3, h4, p, span, div');
        
        textElements.forEach((el, index) => {
            if (index > 10) return; // Limit elements
            if (!el.textContent.trim()) return;
            
            // Skip if element contains other elements (not leaf node)
            if (el.children.length > 0 && el.tagName !== 'P') return;
            
            const text = document.createElementNS(this.xmlns, 'text');
            text.setAttribute('x', '20');
            text.setAttribute('y', yOffset);
            text.setAttribute('font-family', this.defaultFont);
            
            // Set font size based on element type
            let fontSize = '14';
            let fontWeight = 'normal';
            if (el.tagName === 'H1') { fontSize = '24'; fontWeight = 'bold'; }
            else if (el.tagName === 'H2') { fontSize = '20'; fontWeight = 'bold'; }
            else if (el.tagName === 'H3') { fontSize = '18'; fontWeight = 'bold'; }
            else if (el.tagName === 'H4') { fontSize = '16'; fontWeight = 'bold'; }
            
            text.setAttribute('font-size', fontSize);
            text.setAttribute('font-weight', fontWeight);
            text.setAttribute('fill', '#374151');
            
            let content = el.textContent.trim();
            if (content.length > 100) content = content.substring(0, 97) + '...';
            text.textContent = content;
            
            svg.appendChild(text);
            yOffset += parseInt(fontSize) + 10;
            
            if (yOffset > height - 40) return;
        });
        
        return new XMLSerializer().serializeToString(svg);
    }

    /**
     * Wrap text into lines
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
        return lines;
    }

    /**
     * Main export function
     */
    exportAsPureSVG(element, componentType) {
        switch(componentType) {
            case 'data-table':
            case 'section-callout':
                if (element.querySelector('table')) {
                    return this.exportTableAsPureSVG(element);
                }
                return this.exportCalloutAsPureSVG(element);
                
            case 'callout-inline':
                return this.exportCalloutAsPureSVG(element);
                
            case 'phenotype-cards':
                // Use the existing phenotype exporter which is already pure SVG
                const basicExporter = new SVGExporter();
                return basicExporter.exportPhenotypesAsSVG(element);
                
            default:
                return this.exportGenericAsPureSVG(element);
        }
    }

    /**
     * Download SVG
     */
    downloadSVG(svgString, filename = 'export.svg') {
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
window.PureSVGExporter = PureSVGExporter;