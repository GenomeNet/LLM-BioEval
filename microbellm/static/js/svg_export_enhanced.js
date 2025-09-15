/**
 * Enhanced SVG Export with Font Options
 * Provides multiple export formats with different font strategies
 */

class EnhancedSVGExporter extends SVGExporter {
    constructor() {
        super();
        this.fontOptions = {
            'illustrator': {
                family: 'Arial, Helvetica, sans-serif',
                name: 'Illustrator Compatible'
            },
            'academic': {
                family: 'Times New Roman, Times, serif',
                name: 'Academic (Times)'
            },
            'modern': {
                family: 'Helvetica Neue, Helvetica, Arial, sans-serif',
                name: 'Modern (Helvetica)'
            },
            'technical': {
                family: 'Courier New, Courier, monospace',
                name: 'Technical (Monospace)'
            }
        };
    }

    /**
     * Export with font selection
     */
    exportWithFontOption(element, fontOption = 'illustrator') {
        const font = this.fontOptions[fontOption] || this.fontOptions.illustrator;
        
        if (element.querySelector('.phenotype-card')) {
            return this.exportPhenotypesAsSVGWithFont(element, font.family);
        } else {
            return this.exportToSVG(element, {
                fontFamily: font.family
            });
        }
    }

    /**
     * Export phenotypes with specific font
     */
    exportPhenotypesAsSVGWithFont(element, fontFamily) {
        const cards = element.querySelectorAll('.phenotype-card');
        const config = {
            cardWidth: 340,
            cardHeight: 180,
            padding: 20,
            columns: 3,
            gap: 20,
            fontFamily: fontFamily
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
        
        // Add font definition as a style element
        const style = document.createElementNS(this.xmlns, 'style');
        style.textContent = `
            .svg-title { font-family: ${fontFamily}; font-size: 18px; font-weight: bold; fill: #111827; }
            .svg-type { font-family: ${fontFamily}; font-size: 11px; font-weight: 600; fill: #059669; }
            .svg-desc { font-family: ${fontFamily}; font-size: 14px; fill: #6b7280; }
            .svg-tag { font-family: ${fontFamily}; font-size: 13px; font-weight: 500; fill: #047857; }
        `;
        svg.appendChild(style);
        
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
            titleText.setAttribute('class', 'svg-title');
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
            typeText.setAttribute('class', 'svg-type');
            typeText.setAttribute('text-anchor', 'middle');
            typeText.textContent = type.toUpperCase();
            g.appendChild(typeText);
            
            // Description (wrap text)
            const descLines = this.wrapText(description, 50);
            descLines.forEach((line, i) => {
                const descText = document.createElementNS(this.xmlns, 'text');
                descText.setAttribute('x', '20');
                descText.setAttribute('y', 60 + i * 18);
                descText.setAttribute('class', 'svg-desc');
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
                tagText.setAttribute('class', 'svg-tag');
                tagText.setAttribute('text-anchor', 'middle');
                tagText.textContent = value;
                g.appendChild(tagText);
                
                tagX += tagWidth + 8;
            });
            
            svg.appendChild(g);
        });
        
        return new XMLSerializer().serializeToString(svg);
    }

    /**
     * Create export menu
     */
    createExportMenu() {
        const menu = document.createElement('div');
        menu.className = 'export-font-menu';
        menu.innerHTML = `
            <h4 style="margin: 0 0 12px 0; padding: 12px 16px 8px; border-bottom: 1px solid #e5e7eb; font-size: 14px; color: #374151;">
                Select Font Style
            </h4>
            ${Object.entries(this.fontOptions).map(([key, option]) => `
                <button class="export-menu-item" data-font="${key}">
                    ${option.name}
                    <span style="font-size: 11px; color: #9ca3af; display: block; margin-top: 2px;">
                        ${option.family.split(',')[0]}
                    </span>
                </button>
            `).join('')}
        `;
        return menu;
    }
}

// Make globally available
window.EnhancedSVGExporter = EnhancedSVGExporter;