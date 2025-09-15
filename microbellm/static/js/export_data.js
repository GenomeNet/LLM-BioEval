/**
 * Data Export Utilities
 * Export table data as CSV for analysis
 */

class DataExporter {
    
    /**
     * Extract table data from HTML element
     */
    extractTableData(element) {
        const tables = element.querySelectorAll('table');
        if (tables.length === 0) {
            // Try to extract any structured data
            return this.extractStructuredData(element);
        }
        
        const allData = [];
        
        tables.forEach((table, tableIndex) => {
            const data = [];
            
            // Get headers
            const headers = [];
            const headerCells = table.querySelectorAll('thead th, thead td');
            if (headerCells.length > 0) {
                headerCells.forEach(cell => {
                    headers.push(this.cleanText(cell.textContent));
                });
            } else {
                // Try first row as headers
                const firstRow = table.querySelector('tr');
                if (firstRow) {
                    firstRow.querySelectorAll('th, td').forEach(cell => {
                        headers.push(this.cleanText(cell.textContent));
                    });
                }
            }
            
            if (headers.length > 0) {
                data.push(headers);
            }
            
            // Get data rows
            const rows = table.querySelectorAll('tbody tr');
            rows.forEach(row => {
                const rowData = [];
                const cells = row.querySelectorAll('td, th');
                
                cells.forEach(cell => {
                    // Check for bar charts
                    const barChart = cell.querySelector('.bar-chart-container');
                    if (barChart) {
                        // Extract bar chart data
                        const segments = barChart.querySelectorAll('.bar-segment');
                        const values = [];
                        segments.forEach(segment => {
                            const label = segment.className.replace('bar-segment', '').trim();
                            const value = segment.textContent.trim();
                            values.push(`${label}:${value}`);
                        });
                        rowData.push(values.join(' | '));
                    } else {
                        rowData.push(this.cleanText(cell.textContent));
                    }
                });
                
                if (rowData.length > 0) {
                    data.push(rowData);
                }
            });
            
            if (data.length > 0) {
                if (tableIndex > 0) {
                    allData.push([]); // Empty row between tables
                }
                allData.push(...data);
            }
        });
        
        return allData;
    }
    
    /**
     * Extract structured data from non-table elements
     */
    extractStructuredData(element) {
        const data = [];
        
        // Try to extract from phenotype cards
        const cards = element.querySelectorAll('.phenotype-card');
        if (cards.length > 0) {
            data.push(['Phenotype', 'Type', 'Description', 'Allowed Values']);
            cards.forEach(card => {
                const title = card.querySelector('.phenotype-title')?.textContent || '';
                const type = card.querySelector('.phenotype-type')?.textContent || '';
                const description = card.querySelector('.phenotype-description')?.textContent || '';
                const values = Array.from(card.querySelectorAll('.value-tag'))
                    .map(v => v.textContent.trim())
                    .join(', ');
                data.push([
                    this.cleanText(title),
                    this.cleanText(type),
                    this.cleanText(description),
                    values
                ]);
            });
        }
        
        // Try to extract from definition lists
        const dlElements = element.querySelectorAll('dl');
        dlElements.forEach(dl => {
            const terms = dl.querySelectorAll('dt');
            const definitions = dl.querySelectorAll('dd');
            if (terms.length > 0) {
                if (data.length === 0) {
                    data.push(['Term', 'Definition']);
                }
                terms.forEach((term, i) => {
                    const def = definitions[i] || { textContent: '' };
                    data.push([
                        this.cleanText(term.textContent),
                        this.cleanText(def.textContent)
                    ]);
                });
            }
        });
        
        // Extract from lists with data attributes
        const dataLists = element.querySelectorAll('[data-export-value]');
        if (dataLists.length > 0) {
            dataLists.forEach(item => {
                const label = item.getAttribute('data-export-label') || '';
                const value = item.getAttribute('data-export-value') || item.textContent;
                if (label || value) {
                    data.push([this.cleanText(label), this.cleanText(value)]);
                }
            });
        }
        
        return data;
    }
    
    /**
     * Clean text for CSV export
     */
    cleanText(text) {
        if (!text) return '';
        // Remove extra whitespace and newlines
        text = text.trim().replace(/\s+/g, ' ');
        // Escape quotes for CSV
        if (text.includes('"') || text.includes(',') || text.includes('\n')) {
            text = '"' + text.replace(/"/g, '""') + '"';
        }
        return text;
    }
    
    /**
     * Convert data array to CSV string
     */
    arrayToCSV(data) {
        if (!data || data.length === 0) {
            return '';
        }
        
        return data.map(row => {
            if (Array.isArray(row)) {
                return row.join(',');
            }
            return row;
        }).join('\n');
    }
    
    /**
     * Download CSV file
     */
    downloadCSV(csvContent, filename = 'export.csv') {
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Main export function
     */
    exportAsCSV(element, filename = 'data-export.csv') {
        const data = this.extractTableData(element);
        
        if (data.length === 0) {
            alert('No data found to export. This component may not contain tabular data.');
            return;
        }
        
        const csv = this.arrayToCSV(data);
        this.downloadCSV(csv, filename);
        
        console.log(`Exported ${data.length} rows to ${filename}`);
    }
}

// Make globally available
window.DataExporter = DataExporter;