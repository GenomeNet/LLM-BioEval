/**
 * Visual Export - Captures exactly what's shown in the browser
 * Uses multiple strategies for different export needs
 */

class VisualExporter {
    constructor() {
        this.loadedLibraries = {};
    }

    /**
     * Load external library dynamically
     */
    async loadLibrary(name, url) {
        if (this.loadedLibraries[name]) return;
        
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = url;
            script.onload = () => {
                this.loadedLibraries[name] = true;
                resolve();
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    /**
     * Method 1: Browser Print to PDF
     * This gives the most accurate representation
     */
    exportViaprint(element) {
        // Create a new window with just the component
        const printWindow = window.open('', '_blank');
        
        // Get all stylesheets
        const styles = Array.from(document.querySelectorAll('link[rel="stylesheet"], style'))
            .map(style => style.outerHTML)
            .join('\n');
        
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
                        
                        /* Hide UI elements */
                        .export-controls,
                        .viewer-navigation,
                        .viewer-header { 
                            display: none !important; 
                        }
                        
                        /* Ensure backgrounds print */
                        * {
                            -webkit-print-color-adjust: exact !important;
                            print-color-adjust: exact !important;
                            color-adjust: exact !important;
                        }
                        
                        /* Fix dimensions */
                        .viewer-content-wrapper,
                        .viewer-content {
                            width: 100% !important;
                            max-width: none !important;
                            box-shadow: none !important;
                            border: none !important;
                        }
                        
                        /* Ensure tables don't break */
                        table { page-break-inside: avoid; }
                        tr { page-break-inside: avoid; }
                        
                        /* Maintain colors */
                        .bar-segment { 
                            -webkit-print-color-adjust: exact !important;
                            print-color-adjust: exact !important;
                        }
                        
                        @page {
                            size: A4 landscape;
                            margin: 1cm;
                        }
                    }
                </style>
            </head>
            <body>
                ${element.outerHTML}
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
    }

    /**
     * Method 2: Canvas to High-Res Image
     * Good for exact pixel-perfect capture
     */
    async exportAsHighResImage(element, scale = 2) {
        // Load html2canvas if not already loaded
        if (!window.html2canvas) {
            await this.loadLibrary('html2canvas', 
                'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js');
        }
        
        // Hide export controls temporarily
        const exportControls = element.querySelector('.export-controls');
        if (exportControls) exportControls.style.display = 'none';
        
        try {
            const canvas = await html2canvas(element, {
                scale: scale,
                useCORS: true,
                logging: false,
                backgroundColor: '#ffffff',
                windowWidth: element.scrollWidth,
                windowHeight: element.scrollHeight,
                onclone: (clonedDoc) => {
                    // Ensure styles are applied to clone
                    const clonedElement = clonedDoc.querySelector('#componentContent');
                    if (clonedElement) {
                        clonedElement.style.width = element.scrollWidth + 'px';
                        clonedElement.style.height = element.scrollHeight + 'px';
                    }
                }
            });
            
            // Convert to blob and download
            canvas.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `component-export-${Date.now()}.png`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            }, 'image/png', 1.0);
            
        } finally {
            // Restore export controls
            if (exportControls) exportControls.style.display = '';
        }
    }

    /**
     * Method 3: HTML to PDF with jsPDF
     * Converts the canvas to PDF
     */
    async exportAsPDF(element) {
        // Load libraries
        if (!window.html2canvas) {
            await this.loadLibrary('html2canvas', 
                'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js');
        }
        if (!window.jspdf) {
            await this.loadLibrary('jspdf', 
                'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js');
        }
        
        // Hide export controls
        const exportControls = element.querySelector('.export-controls');
        if (exportControls) exportControls.style.display = 'none';
        
        try {
            // Capture as canvas
            const canvas = await html2canvas(element, {
                scale: 2,
                useCORS: true,
                logging: false,
                backgroundColor: '#ffffff'
            });
            
            // Calculate PDF dimensions
            const imgWidth = 297; // A4 width in mm (landscape)
            const imgHeight = (canvas.height * imgWidth) / canvas.width;
            
            // Create PDF
            const { jsPDF } = window.jspdf;
            const pdf = new jsPDF({
                orientation: imgHeight > imgWidth ? 'portrait' : 'landscape',
                unit: 'mm',
                format: 'a4'
            });
            
            // Add image to PDF
            const imgData = canvas.toDataURL('image/png');
            pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
            
            // Download PDF
            pdf.save(`component-export-${Date.now()}.pdf`);
            
        } finally {
            // Restore export controls
            if (exportControls) exportControls.style.display = '';
        }
    }

    /**
     * Method 4: Copy to Clipboard as Image
     * For quick paste into documents
     */
    async copyAsImage(element) {
        if (!window.html2canvas) {
            await this.loadLibrary('html2canvas', 
                'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js');
        }
        
        const exportControls = element.querySelector('.export-controls');
        if (exportControls) exportControls.style.display = 'none';
        
        try {
            const canvas = await html2canvas(element, {
                scale: 2,
                useCORS: true,
                logging: false,
                backgroundColor: '#ffffff'
            });
            
            // Convert canvas to blob
            canvas.toBlob(async (blob) => {
                try {
                    // Use Clipboard API
                    await navigator.clipboard.write([
                        new ClipboardItem({
                            'image/png': blob
                        })
                    ]);
                    alert('Component copied to clipboard! You can paste it into any document.');
                } catch (err) {
                    console.error('Failed to copy:', err);
                    alert('Copy failed. Your browser may not support clipboard access.');
                }
            }, 'image/png');
            
        } finally {
            if (exportControls) exportControls.style.display = '';
        }
    }
}

// Make globally available
window.VisualExporter = VisualExporter;