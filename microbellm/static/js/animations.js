/**
 * Shared animation functions for MicrobeLLM project cards and hero headers
 * Three animation types: bacteria, dna, growth
 */

// Bacteria animation for phenotype analysis
function initBacteriaAnimation(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    let width, height;
    let bacteria = [];
    let animationId;
    let resizeTimeout;
    
    const colors = [
        { r: 34, g: 197, b: 94 },    // Green
        { r: 16, g: 185, b: 129 },   // Emerald
    ];
    
    function resizeCanvas() {
        width = canvas.width = canvas.offsetWidth;
        height = canvas.height = canvas.offsetHeight;
        initBacteria();
    }
    
    function initBacteria() {
        bacteria = [];
        const numBacteria = 35;
        
        for (let i = 0; i < numBacteria; i++) {
            bacteria.push({
                x: Math.random() * width,
                y: Math.random() * height,
                length: 15 + Math.random() * 35,
                width: 4 + Math.random() * 8,
                angle: Math.random() * Math.PI * 2,
                color: colors[Math.floor(Math.random() * colors.length)],
                speed: 0.08 + Math.random() * 0.25,
                rotationSpeed: (Math.random() - 0.5) * 0.015,
                shape: Math.random() > 0.3 ? 'bacillus' : 'coccus'
            });
        }
    }
    
    function drawBacterium(bacterium) {
        ctx.save();
        ctx.translate(bacterium.x, bacterium.y);
        ctx.rotate(bacterium.angle);
        
        ctx.fillStyle = `rgba(${bacterium.color.r}, ${bacterium.color.g}, ${bacterium.color.b}, 0.6)`;
        
        if (bacterium.shape === 'bacillus') {
            ctx.beginPath();
            ctx.moveTo(-bacterium.length/2 + bacterium.width/2, 0);
            ctx.lineTo(bacterium.length/2 - bacterium.width/2, 0);
            ctx.arc(bacterium.length/2 - bacterium.width/2, 0, bacterium.width/2, -Math.PI/2, Math.PI/2);
            ctx.lineTo(-bacterium.length/2 + bacterium.width/2, bacterium.width);
            ctx.arc(-bacterium.length/2 + bacterium.width/2, bacterium.width/2, bacterium.width/2, Math.PI/2, Math.PI * 1.5);
            ctx.closePath();
            ctx.fill();
        } else {
            ctx.beginPath();
            ctx.arc(0, 0, bacterium.width, 0, Math.PI * 2);
            ctx.fill();
        }
        
        ctx.restore();
    }
    
    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        bacteria.forEach(bacterium => {
            bacterium.x += Math.cos(bacterium.angle) * bacterium.speed;
            bacterium.y += Math.sin(bacterium.angle) * bacterium.speed;
            bacterium.angle += bacterium.rotationSpeed;
            
            if (bacterium.x < -50) bacterium.x = width + 50;
            if (bacterium.x > width + 50) bacterium.x = -50;
            if (bacterium.y < -50) bacterium.y = height + 50;
            if (bacterium.y > height + 50) bacterium.y = -50;
            
            drawBacterium(bacterium);
        });
        
        animationId = requestAnimationFrame(animate);
    }
    
    resizeCanvas();
    animate();
    
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(resizeCanvas, 250);
    });
    
    // Return cleanup function
    return () => {
        if (animationId) cancelAnimationFrame(animationId);
        window.removeEventListener('resize', resizeCanvas);
    };
}

// DNA animation for knowledge calibration  
function initDNAAnimation(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    let width, height;
    let columns = [];
    let animationId;
    let resizeTimeout;
    
    const colors = [
        { r: 99, g: 102, b: 241 },   // Blue
        { r: 168, g: 85, b: 247 },   // Purple
        { r: 59, g: 130, b: 246 },   // Light Blue
    ];
    
    function resizeCanvas() {
        width = canvas.width = canvas.offsetWidth;
        height = canvas.height = canvas.offsetHeight;
        initColumns();
    }
    
    function initColumns() {
        columns = [];
        const columnWidth = 8;
        const gap = 2;
        const totalWidth = columnWidth + gap;
        const numColumns = Math.ceil(width / totalWidth);
        
        for (let i = 0; i < numColumns; i++) {
            const segments = [];
            const numSegments = 30;
            
            for (let j = 0; j < numSegments; j++) {
                segments.push({
                    height: 3 + Math.random() * 12,
                    color: colors[Math.floor(Math.random() * colors.length)],
                    offset: Math.random() * height
                });
            }
            
            columns.push({
                x: i * totalWidth,
                segments: segments,
                speed: 0.5 + Math.random() * 0.5
            });
        }
    }
    
    let frame = 0;
    function animate() {
        if (frame % 4 === 0) {  // Update every 4th frame for slower animation
            ctx.clearRect(0, 0, width, height);
            
            columns.forEach(column => {
                column.segments.forEach((segment, index) => {
                    const y = (segment.offset + frame * column.speed * 0.3) % height;
                    
                    ctx.fillStyle = `rgba(${segment.color.r}, ${segment.color.g}, ${segment.color.b}, 0.4)`;
                    ctx.fillRect(column.x, y, 8, segment.height);
                });
            });
        }
        
        frame++;
        animationId = requestAnimationFrame(animate);
    }
    
    resizeCanvas();
    animate();
    
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(resizeCanvas, 250);
    });
    
    // Return cleanup function
    return () => {
        if (animationId) cancelAnimationFrame(animationId);
        window.removeEventListener('resize', resizeCanvas);
    };
}

// Cell growth animation for growth conditions
function initGrowthAnimation(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    let width, height;
    let cells = [];
    let animationId;
    let resizeTimeout;
    
    const colors = [
        { r: 250, g: 204, b: 21 },   // Yellow
        { r: 251, g: 146, b: 60 },   // Orange
        { r: 254, g: 215, b: 102 },  // Light Yellow
    ];
    
    function resizeCanvas() {
        width = canvas.width = canvas.offsetWidth;
        height = canvas.height = canvas.offsetHeight;
        initCells();
    }
    
    function initCells() {
        cells = [];
        const numCells = 25;
        
        for (let i = 0; i < numCells; i++) {
            cells.push({
                x: Math.random() * width,
                y: Math.random() * height,
                radius: 6 + Math.random() * 12,
                color: colors[Math.floor(Math.random() * colors.length)],
                vx: (Math.random() - 0.5) * 0.6,
                vy: (Math.random() - 0.5) * 0.6,
                growth: 0.993 + Math.random() * 0.014
            });
        }
    }
    
    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        cells.forEach((cell, index) => {
            cell.x += cell.vx;
            cell.y += cell.vy;
            cell.radius *= cell.growth;
            
            if (cell.radius > 22) {
                // Cell division
                cells.push({
                    x: cell.x + (Math.random() - 0.5) * 20,
                    y: cell.y + (Math.random() - 0.5) * 20,
                    radius: 7 + Math.random() * 3,
                    color: cell.color,
                    vx: (Math.random() - 0.5) * 0.6,
                    vy: (Math.random() - 0.5) * 0.6,
                    growth: 0.993 + Math.random() * 0.014
                });
                cell.radius = 8 + Math.random() * 4;
            }
            
            if (cell.x < 0 || cell.x > width) cell.vx *= -1;
            if (cell.y < 0 || cell.y > height) cell.vy *= -1;
            
            ctx.fillStyle = `rgba(${cell.color.r}, ${cell.color.g}, ${cell.color.b}, 0.7)`;
            ctx.beginPath();
            ctx.arc(cell.x, cell.y, cell.radius, 0, Math.PI * 2);
            ctx.fill();
        });
        
        // Remove cells that are too small
        cells = cells.filter(cell => cell.radius > 5);
        
        // Maintain population
        while (cells.length < 20) {
            cells.push({
                x: Math.random() * width,
                y: Math.random() * height,
                radius: 6 + Math.random() * 8,
                color: colors[Math.floor(Math.random() * colors.length)],
                vx: (Math.random() - 0.5) * 0.6,
                vy: (Math.random() - 0.5) * 0.6,
                growth: 0.993 + Math.random() * 0.014
            });
        }
        
        if (cells.length > 45) {
            cells = cells.slice(0, 45);
        }
        
        animationId = requestAnimationFrame(animate);
    }
    
    resizeCanvas();
    animate();
    
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(resizeCanvas, 250);
    });
    
    // Return cleanup function
    return () => {
        if (animationId) cancelAnimationFrame(animationId);
        window.removeEventListener('resize', resizeCanvas);
    };
}

// Helper function to initialize animation based on type
function initAnimation(canvasId, animationType) {
    switch(animationType) {
        case 'bacteria':
            return initBacteriaAnimation(canvasId);
        case 'dna':
            return initDNAAnimation(canvasId);
        case 'growth':
            return initGrowthAnimation(canvasId);
        default:
            console.warn(`Unknown animation type: ${animationType}`);
            return null;
    }
}

// Export functions for use in other scripts
window.initBacteriaAnimation = initBacteriaAnimation;
window.initDNAAnimation = initDNAAnimation;
window.initGrowthAnimation = initGrowthAnimation;
window.initAnimation = initAnimation;