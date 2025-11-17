// Initialize current time display
function updateCurrentTime() {
    const now = new Date();
    document.getElementById('current-time').textContent = 
        now.toLocaleString('en-US', { 
            weekday: 'short', 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
}

// Update time on load and every minute
updateCurrentTime();
setInterval(updateCurrentTime, 60000);


// Handle model unload
function unloadModel(modelId) {
    if (confirm(`Are you sure you want to unload model ID ${modelId}?`)) {
        htmx.ajax('POST', '/v1/internal/model/unload', {
            headers: {'X-Model-Id': modelId},
            target: document.querySelector(`#model-card-${modelId}`),
            swap: 'outerHTML',
            handler: function(elt, info) {
                // Refresh the entire models section after unload
                htmx.trigger('#loaded-models', 'load');
            }
        });
    }
}

// Handle KV cache cleanup
function cleanupCache() {
    htmx.ajax('POST', '/management/process/clean-up', {
        headers: {'Content-Type': 'application/json'},
        parameters: {timeout: 1},
        target: document.querySelector('.cache-stats'),
        swap: 'innerHTML'
    });
}

// Process KV cache response
htmx.on('htmx:afterRequest', function(evt) {
    if (evt.detail.pathInfo.path === '/management/kv-cache') {
        const response = JSON.parse(evt.detail.xhr.responseText);
        const maxSize = response.max_size || 10;
        const currentSize = response.current_size || 0;
        const usagePercent = Math.min(100, (currentSize / maxSize) * 100);

        let html = `
        <div class="cache-stats">
            <div class="cache-title">
                <span>Current KV Cache Size</span>
                <span class="cache-size">${currentSize.toFixed(2)} GB / ${maxSize} GB</span>
            </div>
            <div class="cache-progress">
                <div class="cache-progress-bar" style="width: ${usagePercent}%"></div>
            </div>
            <div class="cache-actions">
                <button onclick="cleanupCache()" class="btn">Clean Old Caches</button>
            </div>
        </div>
        `;

        document.querySelector('.cache-stats').outerHTML = html;
    }
});

// /management/processes response
function handleProcessManagentResponse(event){
    const xhr = event.detail.xhr;
    let html = '';
    if (xhr.status >= 200 && xhr.status < 300) {
        const jsonData = JSON.parse(xhr.responseText);

        Object.keys(jsonData.processes).forEach((key) => {
            model_id = key;
            model_name = jsonData.processes[key]['model_name'];
            model_type = jsonData.processes[key]['model_type'];
            context_length = jsonData.processes[key]['context_length'];
            auto_unload = jsonData.processes[key]['auto_unload'];
            priority = jsonData.processes[key]['priority'];

            html += `
            <div class="model-card" id="model-card-${model_id}">
                <span class="model-id">ID: ${model_id}</span>
                <span class="model-info">Name: ${model_name}</span>
                <span class="model-info">Type: ${model_type}</span>
                <span class="model-info">Auto Unload: ${auto_unload ? 'Yes' : 'No'}</span>
                <span class="model-info">Priority: ${priority}</span>
                <span class="model-info">CTX: ${context_length}</span>
                <button onclick="unloadModel('${model_id}')" class="btn danger">Unload</button>
            </div>
            `;
        });
    }
    document.getElementById('loaded-models').innerHTML = html
}


// Fetch model list on page load
document.addEventListener('DOMContentLoaded', function() {
    fetch('/v1/internal/model/list')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            const select = document.getElementById('model-select');
            let options = '<option value="">Select a model...</option>';

            data.model_names.forEach(model => {
                options += `<option value="${model}">${model}</option>`;
            });

            select.innerHTML = options;
        })
        .catch(error => {
            console.error('Error loading model list:', error);
            document.getElementById('model-select').innerHTML = '<option>Error loading models</option>';
        });
});

// Handle model loading form submission
document.getElementById('load-model-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const modelId = document.getElementById('model-id').value;
    const modelName = document.getElementById('model-select').value;
    const useKvCache = document.querySelector("[name=use_kv_cache]").checked;
    const autoUnload = document.querySelector("[name=auto_unload]").checked;
    const priority = parseInt(document.getElementById('priority').value);

    const responseArea = document.getElementById('load-response');
    responseArea.className = 'response-area show';
    responseArea.innerHTML = '<div class="alert info">Loading model...</div>';

    try {
        const response = await fetch('/v1/internal/model/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Model-Id': modelId
            },
            body: JSON.stringify({
                llm_model_name: modelName,
                use_kv_cache: useKvCache,
                auto_unload: autoUnload,
                priority: priority
            })
        });

        const data = await response.json();

        if (response.ok) {
            responseArea.innerHTML = `
                <div class="alert success">
                    <strong>Success!</strong> Model loaded successfully (ID: ${modelId})
                </div>
            `;

            // Refresh the models list
            htmx.trigger('#loaded-models', 'load');

            // Clear form
            document.getElementById('model-id').value = parseInt(modelId) + 1;

            // Hide success message after 5 seconds
            setTimeout(() => {
                responseArea.className = 'response-area';
            }, 5000);
        } else {
            throw new Error(data.detail || 'Failed to load model');
        }
    } catch (error) {
        responseArea.innerHTML = `
            <div class="alert error">
                <strong>Error:</strong> ${error.message}
            </div>
        `;

        // Keep error visible longer
        setTimeout(() => {
            if (responseArea.querySelector('.error')) {
                responseArea.className = 'response-area';
            }
        }, 10000);
    }
});