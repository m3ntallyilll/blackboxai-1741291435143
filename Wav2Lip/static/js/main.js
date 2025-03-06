document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const videoDropZone = document.getElementById('videoDropZone');
    const audioDropZone = document.getElementById('audioDropZone');
    const progress = document.getElementById('progress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const result = document.getElementById('result');
    const error = document.getElementById('error');
    const toggleSettings = document.getElementById('toggleSettings');
    const advancedSettings = document.getElementById('advancedSettings');

    // Toggle advanced settings
    toggleSettings.addEventListener('click', function() {
        advancedSettings.classList.toggle('hidden');
        this.innerHTML = advancedSettings.classList.contains('hidden') ? 
            '<i class="fas fa-cog"></i> Show Settings' : 
            '<i class="fas fa-cog"></i> Hide Settings';
    });

    // Drag and drop handling
    [videoDropZone, audioDropZone].forEach(zone => {
        zone.addEventListener('dragenter', handleDragEnter);
        zone.addEventListener('dragover', handleDragOver);
        zone.addEventListener('dragleave', handleDragLeave);
        zone.addEventListener('drop', handleDrop);
    });

    function handleDragEnter(e) {
        preventDefaults(e);
        this.classList.add('dragging');
    }

    function handleDragOver(e) {
        preventDefaults(e);
        this.classList.add('dragging');
    }

    function handleDragLeave(e) {
        preventDefaults(e);
        this.classList.remove('dragging');
    }

    function handleDrop(e) {
        preventDefaults(e);
        this.classList.remove('dragging');

        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length) {
            const file = files[0];
            const inputId = this.id === 'videoDropZone' ? 'faceVideo' : 'audioFile';
            const input = document.getElementById(inputId);
            input.files = files;
            input.dispatchEvent(new Event('change'));
        }
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // File input preview
    document.getElementById('faceVideo').addEventListener('change', function() {
        handleFileSelect(this, 'videoPreview', 'video');
    });

    document.getElementById('audioFile').addEventListener('change', function() {
        handleFileSelect(this, 'audioPreview', 'audio');
    });

    function handleFileSelect(input, previewId, mediaType) {
        const file = input.files[0];
        if (!file) return;

        const preview = document.getElementById(previewId);
        const media = preview.querySelector(mediaType);
        const info = preview.querySelector('p');

        media.src = URL.createObjectURL(file);
        info.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
        preview.classList.remove('hidden');
    }

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Hide previous results/errors
        result.classList.add('hidden');
        error.classList.add('hidden');
        
        // Show progress
        progress.classList.remove('hidden');
        progressBar.style.width = '0%';
        progressText.textContent = '0%';

        const formData = new FormData(this);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                // Update progress to complete
                progressBar.style.width = '100%';
                progressText.textContent = '100%';

                // Show result
                result.classList.remove('hidden');
                const video = document.getElementById('resultVideo');
                video.src = `/results/${data.result_path}`;
                
                const downloadBtn = document.getElementById('downloadBtn');
                downloadBtn.href = `/results/${data.result_path}`;
                
                // Hide progress after a delay
                setTimeout(() => {
                    progress.classList.add('hidden');
                }, 1000);
            } else {
                if (data.type === 'demo_mode') {
                    throw new Error('Lip sync functionality is not available in demo mode. Please install the required dependencies.');
                } else {
                    throw new Error(data.error || 'An error occurred');
                }
            }
        } catch (error) {
            // Show error
            progress.classList.add('hidden');
            const errorDiv = document.getElementById('error');
            const errorText = document.getElementById('errorText');
            errorText.innerHTML = error.message;
            if (error.message.includes('demo mode')) {
                errorText.innerHTML += '<div class="mt-2"><pre class="text-sm bg-gray-800 text-white p-2 rounded">pip install torch torchvision torchaudio face-alignment</pre></div>';
            }
            errorDiv.classList.remove('hidden');
            errorDiv.classList.add('error-shake');
            setTimeout(() => errorDiv.classList.remove('error-shake'), 500);
        }
    });

    // Retry button
    document.getElementById('retryBtn').addEventListener('click', function() {
        result.classList.add('hidden');
        error.classList.add('hidden');
        form.reset();
        document.getElementById('videoPreview').classList.add('hidden');
        document.getElementById('audioPreview').classList.add('hidden');
    });

    // Parameter tooltips
    document.querySelectorAll('[data-tooltip]').forEach(element => {
        const tooltip = document.createElement('span');
        tooltip.className = 'tooltiptext';
        tooltip.textContent = element.dataset.tooltip;
        element.appendChild(tooltip);
    });
});
