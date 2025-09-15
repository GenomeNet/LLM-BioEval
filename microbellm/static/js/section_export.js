/**
 * Section Export Functionality
 * Adds export buttons to section-callout elements on research pages
 */

class SectionExporter {
    constructor() {
        this.init();
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.addExportButtons());
        } else {
            this.addExportButtons();
        }
    }

    addExportButtons() {
        // Find all section-callout elements
        const sections = document.querySelectorAll('.section-callout');
        
        sections.forEach((section, index) => {
            // Skip if buttons already added
            if (section.querySelector('.section-export-controls')) return;
            
            // Create export controls container
            const exportControls = document.createElement('div');
            exportControls.className = 'section-export-controls';
            exportControls.innerHTML = `
                <button class="section-export-btn" onclick="sectionExporter.exportSectionPDF(this)" title="Save as PDF">
                    <svg viewBox="0 0 16 16" fill="currentColor">
                        <path d="M14 4.5V14a2 2 0 0 1-2 2h-1v-1h1a1 1 0 0 0 1-1V4.5h-2A1.5 1.5 0 0 1 9.5 3V1H4a1 1 0 0 0-1 1v9H2V2a2 2 0 0 1 2-2h5.5L14 4.5ZM1.6 11.85H0v3.999h.791v-1.342h.803c.287 0 .531-.057.732-.173.203-.117.358-.275.463-.474a1.42 1.42 0 0 0 .161-.677c0-.25-.053-.476-.158-.677a1.176 1.176 0 0 0-.46-.477c-.2-.12-.443-.179-.732-.179Zm.545 1.333a.795.795 0 0 1-.085.38.574.574 0 0 1-.238.241.794.794 0 0 1-.375.082H.788V12.48h.66c.218 0 .389.06.512.181.123.122.185.296.185.522Zm1.217-1.333v3.999h1.46c.401 0 .734-.08.998-.237a1.45 1.45 0 0 0 .595-.689c.13-.3.196-.662.196-1.084 0-.42-.065-.778-.196-1.075a1.426 1.426 0 0 0-.589-.68c-.264-.156-.599-.234-1.005-.234H3.362Zm.791.645h.563c.248 0 .45.05.609.152a.89.89 0 0 1 .354.454c.079.201.118.452.118.753a2.3 2.3 0 0 1-.068.592 1.14 1.14 0 0 1-.196.422.8.8 0 0 1-.334.252 1.298 1.298 0 0 1-.483.082h-.563v-2.707Zm3.743 1.763v1.591h-.79V11.85h2.548v.653H7.896v1.117h1.606v.638H7.896Z"/>
                    </svg>
                    PDF
                </button>
                <button class="section-export-btn" onclick="sectionExporter.exportSectionCSV(this)" title="Export data as CSV">
                    <svg viewBox="0 0 16 16" fill="currentColor">
                        <path d="M14 14V4.5L9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2zM9.5 3A1.5 1.5 0 0 0 11 4.5h2V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1h5.5v2z"/>
                        <path d="M5.5 11.5a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5H6a.5.5 0 0 1-.5-.5v-1zm0-3a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5H6a.5.5 0 0 1-.5-.5v-1zm0-3a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5H6a.5.5 0 0 1-.5-.5v-1zm3 6a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5H9a.5.5 0 0 1-.5-.5v-1zm0-3a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5H9a.5.5 0 0 1-.5-.5v-1zm0-3a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5H9a.5.5 0 0 1-.5-.5v-1z"/>
                    </svg>
                    CSV
                </button>
            `;
            
            // Insert at the beginning of the section
            section.insertBefore(exportControls, section.firstChild);
        });
    }

    async exportSectionPDF(button) {
        const section = button.closest('.section-callout');
        if (!section) return;
        
        // Hide export controls temporarily
        const controls = section.querySelector('.section-export-controls');
        if (controls) controls.style.display = 'none';
        
        try {
            // Create a new window with just the section content
            const printWindow = window.open('', '_blank');
            
            // Get all stylesheets
            const styles = Array.from(document.querySelectorAll('link[rel="stylesheet"], style'))
                .map(style => style.outerHTML)
                .join('\n');
            
            // Clone the section to modify it
            const sectionClone = section.cloneNode(true);
            
            // Remove export controls from clone
            const clonedControls = sectionClone.querySelector('.section-export-controls');
            if (clonedControls) clonedControls.remove();
            
            // Build the print document
            const printContent = `
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Export - ${document.title}</title>
                    ${styles}
                    <style>
                        @media print {
                            body { 
                                margin: 0; 
                                padding: 20px;
                                -webkit-print-color-adjust: exact !important;
                                print-color-adjust: exact !important;
                                color-adjust: exact !important;
                            }
                            
                            /* Ensure backgrounds print */
                            * {
                                -webkit-print-color-adjust: exact !important;
                                print-color-adjust: exact !important;
                                color-adjust: exact !important;
                            }
                            
                            /* Reset section positioning for print */
                            .section-callout {
                                position: static !important;
                                width: 100% !important;
                                left: auto !important;
                                transform: none !important;
                                margin: 0 !important;
                            }
                            
                            .section-callout__content {
                                max-width: none !important;
                                padding: 20px !important;
                            }
                            
                            /* Ensure tables don't break */
                            table { page-break-inside: avoid; }
                            tr { page-break-inside: avoid; }
                            
                            @page {
                                size: A4 landscape;
                                margin: 1cm;
                            }
                        }
                        
                        /* Screen styles for preview */
                        @media screen {
                            body {
                                margin: 0;
                                padding: 20px;
                                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                            }
                            
                            .section-callout {
                                position: static !important;
                                width: 100% !important;
                                left: auto !important;
                                transform: none !important;
                                margin: 0 !important;
                            }
                        }
                    </style>
                </head>
                <body>
                    ${sectionClone.outerHTML}
                </body>
                </html>
            `;
            
            printWindow.document.write(printContent);
            printWindow.document.close();
            
            // Wait for content to load then print
            printWindow.onload = function() {
                setTimeout(() => {
                    printWindow.print();
                    // Don't close immediately - let user save as PDF
                }, 500);
            };
            
        } catch (error) {
            console.error('Export failed:', error);
            alert('Failed to export PDF. Please try again.');
        } finally {
            // Restore export controls
            if (controls) controls.style.display = '';
        }
    }

    async exportSectionCSV(button) {
        const section = button.closest('.section-callout');
        if (!section) return;
        
        try {
            // Load DataExporter if not already loaded
            if (!window.DataExporter) {
                await this.loadScript('/static/js/export_data.js');
            }
            
            const exporter = new DataExporter();
            
            // Get section title for filename
            const titleElement = section.querySelector('.section-callout__title');
            const title = titleElement ? titleElement.textContent.trim() : 'section';
            const filename = `${title.toLowerCase().replace(/\s+/g, '-')}-data.csv`;
            
            exporter.exportAsCSV(section, filename);
            
        } catch (error) {
            console.error('Export failed:', error);
            alert('Failed to export data. The section may not contain tabular data.');
        }
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            if (document.querySelector(`script[src="${src}"]`)) {
                resolve();
                return;
            }
            
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }
}

// Initialize and make globally available
const sectionExporter = new SectionExporter();
window.sectionExporter = sectionExporter;