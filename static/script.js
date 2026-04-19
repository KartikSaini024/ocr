// Global to hold latest JSON data
let lastExtractedData = null;
let pdlMapping = {};
let pdlValueMapping = {};
let startTime = null;
let timerInterval = null;

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loadingState = document.getElementById('loading');
    const uploadSection = document.getElementById('upload-section');
    const resultsSection = document.getElementById('results-section');
    const resultsGrid = document.getElementById('results-grid');
    const resetBtn = document.getElementById('reset-btn');
    const summaryCard = document.getElementById('summary-card');
    const extractionStats = document.getElementById('extraction-stats');
    const confidenceContainer = document.getElementById('confidence-container');
    const confidenceBadge = document.getElementById('confidence-badge');

    // Load PDL mapping
    async function initMapping() {
        try {
            const res = await fetch('/api/mapping');
            const data = await res.json();
            pdlMapping = data.mapping || {};
            pdlValueMapping = data.values || {};
        } catch(e) {
            console.error('Failed to load mapping:', e);
        }
    }
    initMapping();

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type === 'application/pdf') {
                uploadFile(file);
            } else {
                alert('Please upload a PDF file.');
            }
        }
    }

    function startTimer() {
        const timerDisplay = document.getElementById('elapsed-timer');
        startTime = Date.now();
        if (timerInterval) clearInterval(timerInterval);
        
        timerInterval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            timerDisplay.textContent = formatTime(elapsed);
        }, 100);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function formatTime(ms) {
        const totalSeconds = Math.floor(ms / 1000);
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        const milliseconds = Math.floor((ms % 1000) / 100);
        
        return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds}`;
    }

    async function uploadFile(file) {
        // UI transitions
        dropZone.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingState.classList.remove('hidden');
        summaryCard.style.display = 'none';
        
        // Start timer
        startTimer();
        
        // Clear previous logs
        const logsContent = document.getElementById('logs-content');
        logsContent.innerHTML = '';
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Upload failed');
            }

            const jobId = result.job_id;
            console.log(`Job started: ${jobId}`);
            
            // Connect to SSE for progress updates
            trackProgress(jobId);

        } catch (error) {
            alert(`Error: ${error.message}`);
            // Reset UI
            loadingState.classList.add('hidden');
            dropZone.classList.remove('hidden');
            stopTimer();
        }
    }

    function trackProgress(jobId) {
        const logsContent = document.getElementById('logs-content');
        const eventSource = new EventSource(`/api/progress/${jobId}`);

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.status === 'processing') {
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                logEntry.textContent = `> ${data.message}`;
                logsContent.appendChild(logEntry);
                
                // Auto-scroll
                logsContent.scrollTop = logsContent.scrollHeight;
            } 
            else if (data.status === 'completed') {
                eventSource.close();
                fetchResults(jobId);
            }
            else if (data.status === 'failed') {
                eventSource.close();
                alert(`Processing failed: ${data.message}`);
                loadingState.classList.add('hidden');
                dropZone.classList.remove('hidden');
                stopTimer();
            }
        };

        eventSource.onerror = (error) => {
            console.error('SSE Error:', error);
            eventSource.close();
            // If SSE fails, we can fall back to polling or just wait
            // For now, let's try to fetch results anyway after a delay
            setTimeout(() => fetchResults(jobId), 2000);
        };
    }

    async function fetchResults(jobId) {
        try {
            const response = await fetch(`/api/result/${jobId}`);
            const result = await response.json();

            if (!response.ok) {
                if (response.status === 404 && result.status === 'processing') {
                    // Still processing, poll again
                    setTimeout(() => fetchResults(jobId), 2000);
                    return;
                }
                throw new Error(result.error || 'Failed to fetch results');
            }

            lastExtractedData = result.data;
            displayResults(result.data);

        } catch (error) {
            console.error('Fetch Results Error:', error);
            // Don't alert immediately, might be transient
            setTimeout(() => fetchResults(jobId), 3000);
        }
    }

    function displayResults(data) {
        resultsGrid.innerHTML = '';
        
        let displayData = data;
        let mappedCount = 0;
        let isError = false;
        
        if (!data) return;

        // Handle error cases where raw_text is returned
        if (data.error && data.raw_text) {
            isError = true;
            displayData = { 'Raw Text Extracted (Mapping Failed)': data.raw_text };
        } else {
            // Count actual mapped keys, excluding raw_text and processing_confidence
            mappedCount = Object.keys(data).filter(k => k !== 'raw_text' && k !== 'processing_confidence').length;
            
            // Extract and display processing confidence
            if (data.processing_confidence !== undefined) {
                const confScore = parseInt(data.processing_confidence);
                if (!isNaN(confScore)) {
                    confidenceBadge.textContent = `${confScore}%`;
                    confidenceBadge.className = 'confidence-badge';
                    if (confScore >= 80) confidenceBadge.classList.add('conf-high');
                    else if (confScore >= 50) confidenceBadge.classList.add('conf-med');
                    else confidenceBadge.classList.add('conf-low');
                    
                    confidenceContainer.style.display = 'flex';
                }
                
                // Remove from displayData so it doesn't render as a card
                delete displayData.processing_confidence;
            }
        }

        // Generate cards for each key-value pair
        for (const [key, value] of Object.entries(displayData)) {
            // Skip empty values or raw text if we successfully mapped
            if (!value) continue;
            if (key === 'raw_text' && !isError) continue;
            
            const isRawText = key.includes('Raw Text');
            const fieldName = pdlMapping[key] ? `${key} - ${pdlMapping[key]}` : key;
            
            const card = document.createElement('div');
            card.className = isRawText ? 'result-card raw-text-card' : 'result-card';
            
            let displayValue = value;
            if (pdlValueMapping[key] && pdlValueMapping[key][String(value)]) {
                displayValue = `${value} (${pdlValueMapping[key][String(value)]})`;
            }

            if (typeof value === 'object') {
                displayValue = `<pre>${JSON.stringify(value, null, 2)}</pre>`;
            } else if (value && String(value).length > 200 && !isRawText) {
                // Truncate very long text with scroll for normal fields
                displayValue = `<div style="max-height: 150px; overflow-y: auto; padding-right: 8px;">${value}</div>`;
            } else if (isRawText) {
                displayValue = `<pre>${value.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>`;
            }

            card.innerHTML = `
                <div class="result-key">${isRawText ? key : `PDL: ${fieldName}`}</div>
                <div class="result-value">${displayValue}</div>
            `;
            
            resultsGrid.appendChild(card);
        }

        // Update summary
        if (!isError) {
            extractionStats.textContent = `Successfully processed document and mapped ${mappedCount} fields to PDL.`;
            summaryCard.style.display = 'flex';
        }

        loadingState.classList.add('hidden');
        uploadSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        stopTimer();
    }

    resetBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        dropZone.classList.remove('hidden');
        confidenceContainer.style.display = 'none';
        fileInput.value = '';
        lastExtractedData = null;
    });
});

// Export Global Function
window.exportJSON = function() {
    if (!lastExtractedData) return;
    
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastExtractedData, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "extracted_medical_data.json");
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}
