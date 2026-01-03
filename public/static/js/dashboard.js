/**
 * IoT DDoS Detection Dashboard - JavaScript
 * ==========================================
 * Handles all frontend logic for the advanced dashboard
 * 
 * Author: IoT Security Research Team
 * Date: 2026-01-03
 */

// =============================================================================
// GLOBAL VARIABLES
// =============================================================================

const socket = io(window.location.origin);
let currentTab = 'dashboard';
let liveChart = null;
let accuracyChart = null;
let errorRatesChart = null;
let improvementChart = null;
let modelMetrics = {};
let trainingPollingInterval = null;

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    initializeSocket();
    loadInitialData();
    updateClock();
    setInterval(updateClock, 1000);
});

// Model colors for dynamic chart updates
const MODEL_COLORS = {
    'CNN': { border: '#3b82f6', bg: 'rgba(59,130,246,0.1)' },
    'LSTM': { border: '#22c55e', bg: 'rgba(34,197,94,0.1)' },
    'Hybrid': { border: '#a855f7', bg: 'rgba(168,85,247,0.1)' },
    'Parallel': { border: '#f97316', bg: 'rgba(249,115,22,0.1)' },
    'default': { border: '#6b7280', bg: 'rgba(107,114,128,0.1)' }
};

// Store loaded model names for chart updates
let loadedModelNames = [];

function initializeCharts() {
    // Live Chart for Real-time tab - will be updated dynamically when models load
    const liveCtx = document.getElementById('liveChart')?.getContext('2d');
    if (liveCtx) {
        liveChart = new Chart(liveCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [] // Will be populated dynamically
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    y: { beginAtZero: true, max: 1, grid: { color: '#334155' } },
                    x: { grid: { color: '#334155' } }
                },
                plugins: { legend: { labels: { color: '#94a3b8' } } }
            }
        });
    }
    
    // Accuracy Chart for Dashboard
    const accCtx = document.getElementById('accuracyChart')?.getContext('2d');
    if (accCtx) {
        accuracyChart = new Chart(accCtx, {
            type: 'bar',
            data: {
                labels: ['CNN', 'LSTM', 'Hybrid'],
                datasets: [{
                    label: 'Accuracy (%)',
                    data: [0, 0, 0],
                    backgroundColor: ['#3b82f6', '#22c55e', '#a855f7'],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100, grid: { color: '#334155' } },
                    x: { grid: { display: false } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
    
    // Error Rates Chart for Dashboard
    const errCtx = document.getElementById('errorRatesChart')?.getContext('2d');
    if (errCtx) {
        errorRatesChart = new Chart(errCtx, {
            type: 'bar',
            data: {
                labels: ['CNN', 'LSTM', 'Hybrid'],
                datasets: [
                    { label: 'FPR (%)', data: [0, 0, 0], backgroundColor: '#ef4444', borderRadius: 8 },
                    { label: 'FNR (%)', data: [0, 0, 0], backgroundColor: '#f97316', borderRadius: 8 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, grid: { color: '#334155' } },
                    x: { grid: { display: false } }
                },
                plugins: { legend: { labels: { color: '#94a3b8' } } }
            }
        });
    }
}

function initializeSocket() {
    socket.on('connect', () => {
        console.log('✅ Connected to server');
        document.getElementById('systemStatus').classList.add('pulse-green');
    });
    
    socket.on('disconnect', () => {
        console.log('❌ Disconnected');
        document.getElementById('systemStatus').classList.remove('pulse-green');
    });
    
    socket.on('traffic_update', (data) => {
        updateLiveMonitor(data);
    });
}

async function loadInitialData() {
    try {
        // Load system info
        const sysRes = await fetch('/api/system/info');
        const sysData = await sysRes.json();
        if (sysData.status === 'ok') {
            document.getElementById('systemDevice').textContent = sysData.system.gpu_available 
                ? `GPU: ${sysData.system.gpu_info?.name || 'CUDA'}` 
                : 'CPU Mode';
        }
        
        // Load models info
        await loadModelsData();
        
        // Load dataset info
        await loadDatasetInfo();
        
        // Load history
        await loadHistory();
        
        // Load compare data
        await loadCompareData();
        
    } catch (err) {
        console.error('Error loading initial data:', err);
        showToast('Error loading data', 'error');
    }
}

// =============================================================================
// DATA LOADING FUNCTIONS
// =============================================================================

async function loadModelsData() {
    try {
        const res = await fetch('/api/models/list');
        const data = await res.json();
        
        if (data.status === 'ok') {
            const models = data.models.filter(m => m.exists);
            document.getElementById('statModels').textContent = models.length;
            
            // Find best metrics
            let bestAcc = 0, lowestFPR = 100;
            models.forEach(m => {
                if (m.metrics) {
                    modelMetrics[m.name] = m.metrics;
                    if (m.metrics.accuracy > bestAcc) bestAcc = m.metrics.accuracy;
                    if (m.metrics.fpr < lowestFPR) lowestFPR = m.metrics.fpr;
                }
            });
            
            document.getElementById('statAccuracy').textContent = bestAcc > 0 ? `${bestAcc.toFixed(2)}%` : '-';
            document.getElementById('statFPR').textContent = lowestFPR < 100 ? `${lowestFPR.toFixed(2)}%` : '-';
            
            // Update ranking table
            updateModelRankingTable(models);
            
            // Update charts
            updateDashboardCharts(models);
            
            // Generate model cards for Models tab
            generateModelCards(models);
            
            // Update detailed metrics table
            updateDetailedMetricsTable(models);
            
            // Update dynamic model checkboxes for Evaluation and Training tabs
            renderModelCheckboxes(data.models, 'evalModelsContainer', 'eval-model-cb');
            renderModelCheckboxes(data.models, 'trainModelsContainer', 'train-model-cb');
            
            // Update live chart datasets for all loaded models
            updateLiveChartDatasets(models);
        }
    } catch (err) {
        console.error('Error loading models:', err);
    }
}

// Function to update live chart datasets dynamically based on loaded models
function updateLiveChartDatasets(models) {
    if (!liveChart) return;
    
    const availableModels = models.filter(m => m.exists);
    loadedModelNames = availableModels.map(m => m.name);
    
    // Create datasets for each model
    liveChart.data.datasets = availableModels.map(m => {
        const colors = MODEL_COLORS[m.name] || MODEL_COLORS['default'];
        return {
            label: m.name,
            data: [],
            borderColor: colors.border,
            backgroundColor: colors.bg,
            tension: 0.4,
            borderWidth: 2,
            pointRadius: 0
        };
    });
    
    liveChart.update();
}

// Function to dynamically render model checkboxes
function renderModelCheckboxes(models, containerId, checkboxClass) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const colors = {
        'CNN': 'blue',
        'LSTM': 'green', 
        'Hybrid': 'purple',
        'Parallel': 'orange'
    };
    
    // Only show models that exist
    const availableModels = models.filter(m => m.exists);
    
    if (availableModels.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-sm">No models available. Train some models first.</p>';
        return;
    }
    
    if (checkboxClass === 'eval-model-cb') {
        // Horizontal layout for evaluation tab
        container.innerHTML = availableModels.map(m => {
            const color = colors[m.name] || 'gray';
            return `
            <label class="flex items-center space-x-2 cursor-pointer bg-slate-700 px-4 py-2 rounded-lg hover:bg-slate-600 transition">
                <input type="checkbox" value="${m.name}" class="${checkboxClass} accent-${color}-500" checked>
                <span class="text-${color}-400 font-medium">${m.name}</span>
            </label>`;
        }).join('');
    } else {
        // Vertical layout for training tab
        container.innerHTML = availableModels.map(m => {
            const color = colors[m.name] || 'gray';
            return `
            <label class="flex items-center space-x-3 cursor-pointer p-2 hover:bg-slate-700 rounded transition">
                <input type="checkbox" value="${m.name}" class="${checkboxClass} accent-${color}-500" checked>
                <span class="text-${color}-400">${m.name}</span>
            </label>`;
        }).join('');
    }
}

function updateModelRankingTable(models) {
    const tbody = document.getElementById('modelRankingTable');
    if (!tbody) return;
    
    // Sort by FPR (lower is better)
    const sorted = models.filter(m => m.metrics).sort((a, b) => a.metrics.fpr - b.metrics.fpr);
    
    if (sorted.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-gray-500 py-8">No evaluation data available. Run evaluation first.</td></tr>';
        return;
    }
    
    const rankings = ['🥇', '🥈', '🥉'];
    const rankingClasses = ['ranking-gold', 'ranking-silver', 'ranking-bronze'];
    
    tbody.innerHTML = sorted.map((m, idx) => `
        <tr class="border-b border-slate-700">
            <td class="font-bold">${m.name}</td>
            <td>${m.metrics.accuracy.toFixed(2)}%</td>
            <td class="${m.metrics.fpr < 30 ? 'text-green-400' : m.metrics.fpr < 50 ? 'text-yellow-400' : 'text-red-400'} font-bold">
                ${m.metrics.fpr.toFixed(2)}%
            </td>
            <td>${m.metrics.fnr.toFixed(2)}%</td>
            <td>${m.metrics.roc_auc.toFixed(4)}</td>
            <td>
                <span class="${rankingClasses[idx] || ''} text-white px-3 py-1 rounded-full text-sm font-bold">
                    ${rankings[idx] || `#${idx + 1}`}
                </span>
            </td>
        </tr>
    `).join('');
}

function updateDashboardCharts(models) {
    const sorted = models.filter(m => m.metrics);
    
    if (accuracyChart) {
        const labels = sorted.map(m => m.name);
        const accData = sorted.map(m => m.metrics.accuracy);
        accuracyChart.data.labels = labels;
        accuracyChart.data.datasets[0].data = accData;
        accuracyChart.update();
    }
    
    if (errorRatesChart) {
        const labels = sorted.map(m => m.name);
        const fprData = sorted.map(m => m.metrics.fpr);
        const fnrData = sorted.map(m => m.metrics.fnr);
        errorRatesChart.data.labels = labels;
        errorRatesChart.data.datasets[0].data = fprData;
        errorRatesChart.data.datasets[1].data = fnrData;
        errorRatesChart.update();
    }
}

function generateModelCards(models) {
    const container = document.getElementById('modelCardsContainer');
    if (!container) return;
    
    const colors = { CNN: 'blue', LSTM: 'green', Hybrid: 'purple', Parallel: 'orange' };
    
    container.innerHTML = models.map(m => {
        const color = colors[m.name] || 'gray';
        const hasMetrics = m.metrics !== null;
        
        return `
        <div class="bg-slate-800 rounded-xl p-6 border border-slate-700 hover:border-${color}-500 transition">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <h4 class="font-bold text-${color}-400 text-lg">${m.name}</h4>
                    <p class="text-xs text-gray-500">${m.filename}</p>
                </div>
                <span class="px-2 py-1 rounded text-xs ${m.exists ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'}">
                    ${m.exists ? 'Loaded' : 'Missing'}
                </span>
            </div>
            ${hasMetrics ? `
            <div class="space-y-2">
                <div class="flex justify-between text-sm">
                    <span class="text-gray-400">Accuracy</span>
                    <span class="font-bold text-${color}-400">${m.metrics.accuracy.toFixed(2)}%</span>
                </div>
                <div class="flex justify-between text-sm">
                    <span class="text-gray-400">FPR</span>
                    <span class="font-bold">${m.metrics.fpr.toFixed(2)}%</span>
                </div>
                <div class="flex justify-between text-sm">
                    <span class="text-gray-400">ROC-AUC</span>
                    <span class="font-bold">${m.metrics.roc_auc.toFixed(4)}</span>
                </div>
            </div>
            ` : `
            <div class="text-center py-4 text-gray-500">
                <i class="fas fa-exclamation-circle text-2xl mb-2"></i>
                <p class="text-sm">No evaluation data</p>
            </div>
            `}
            <div class="mt-4 pt-4 border-t border-slate-700 text-xs text-gray-500">
                <p><i class="far fa-hdd mr-1"></i>${m.size_mb} MB</p>
                ${m.modified ? `<p><i class="far fa-clock mr-1"></i>${new Date(m.modified).toLocaleString('vi-VN')}</p>` : ''}
            </div>
        </div>
        `;
    }).join('');
}

function updateDetailedMetricsTable(models) {
    const tbody = document.getElementById('detailedMetricsTable');
    if (!tbody) return;
    
    const sorted = models.filter(m => m.metrics);
    
    if (sorted.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center text-gray-500 py-8">No metrics available</td></tr>';
        return;
    }
    
    tbody.innerHTML = sorted.map(m => `
        <tr class="border-b border-slate-700">
            <td class="font-bold">${m.name}</td>
            <td>${m.metrics.accuracy.toFixed(2)}%</td>
            <td>${m.metrics.precision.toFixed(2)}%</td>
            <td>${m.metrics.recall.toFixed(2)}%</td>
            <td>${m.metrics.f1_score.toFixed(2)}%</td>
            <td class="${m.metrics.fpr < 30 ? 'text-green-400' : 'text-yellow-400'}">${m.metrics.fpr.toFixed(2)}%</td>
            <td>${m.metrics.fnr.toFixed(2)}%</td>
            <td>${m.metrics.roc_auc.toFixed(4)}</td>
        </tr>
    `).join('');
}

async function loadDatasetInfo() {
    try {
        const res = await fetch('/api/dataset/info');
        const data = await res.json();
        
        if (data.status === 'ok' && data.dataset.processed_data) {
            const ds = data.dataset.processed_data;
            document.getElementById('datasetTrainSamples').textContent = ds.train_samples?.toLocaleString() || '-';
            document.getElementById('datasetValSamples').textContent = ds.val_samples?.toLocaleString() || '-';
            document.getElementById('datasetTestSamples').textContent = ds.test_samples?.toLocaleString() || '-';
            document.getElementById('datasetFeatures').textContent = ds.features || '-';
            
            if (ds.class_distribution) {
                document.getElementById('datasetNormalCount').textContent = 
                    (ds.class_distribution['0'] || 0).toLocaleString();
                document.getElementById('datasetAttackCount').textContent = 
                    (ds.class_distribution['1'] || 0).toLocaleString();
            }
        }
    } catch (err) {
        console.error('Error loading dataset info:', err);
    }
}

async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        
        if (data.status === 'ok') {
            // Update training count
            document.getElementById('statTrainings').textContent = data.history.trainings?.length || 0;
            
            // Training history table
            const trainTable = document.getElementById('trainingHistoryTable');
            if (trainTable && data.history.trainings?.length > 0) {
                trainTable.innerHTML = data.history.trainings.reverse().map(t => `
                    <tr class="border-b border-slate-700">
                        <td>#${t.id}</td>
                        <td>${new Date(t.timestamp).toLocaleString('vi-VN')}</td>
                        <td>${t.models?.join(', ') || '-'}</td>
                        <td>${t.epochs || '-'}</td>
                        <td>${t.batch_size || '-'}</td>
                        <td>${t.duration_seconds ? Math.round(t.duration_seconds) + 's' : '-'}</td>
                        <td>
                            <span class="px-2 py-1 rounded text-xs ${t.status === 'completed' ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'}">
                                ${t.status || 'unknown'}
                            </span>
                        </td>
                    </tr>
                `).join('');
            }
            
            // Evaluation history table
            const evalTable = document.getElementById('evaluationHistoryTable');
            if (evalTable && data.history.evaluations?.length > 0) {
                evalTable.innerHTML = data.history.evaluations.reverse().map(e => `
                    <tr class="border-b border-slate-700">
                        <td>#${e.id}</td>
                        <td>${new Date(e.timestamp).toLocaleString('vi-VN')}</td>
                        <td>${e.models?.join(', ') || '-'}</td>
                        <td>
                            <span class="px-2 py-1 rounded text-xs ${e.status === 'completed' ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'}">
                                ${e.status || 'unknown'}
                            </span>
                        </td>
                    </tr>
                `).join('');
            }
        }
    } catch (err) {
        console.error('Error loading history:', err);
    }
}

async function loadCompareData() {
    try {
        const res = await fetch('/api/compare');
        const data = await res.json();
        
        if (data.status === 'ok') {
            renderCompareTable(data.comparison);
            updateImprovementChart(data.comparison.improvements);
        }
    } catch (err) {
        console.error('Error loading compare data:', err);
    }
}

function renderCompareTable(comparison) {
    const container = document.getElementById('compareContainer');
    if (!container) return;
    
    if (!comparison.old || !comparison.new) {
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <i class="fas fa-exchange-alt text-4xl mb-4"></i>
                <p>No comparison data available. Run evaluations first.</p>
            </div>
        `;
        return;
    }
    
    let html = '<div class="overflow-x-auto"><table class="metric-table w-full">';
    html += `
        <thead>
            <tr class="border-b border-slate-700 text-left">
                <th>Model</th>
                <th>Metric</th>
                <th>Old Value</th>
                <th>New Value</th>
                <th>Change</th>
            </tr>
        </thead>
        <tbody>
    `;
    
    for (const model of Object.keys(comparison.new)) {
        const oldM = comparison.old?.[model] || {};
        const newM = comparison.new[model];
        const imp = comparison.improvements?.[model] || {};
        
        const metrics = [
            { name: 'Accuracy', old: oldM.accuracy, new: newM.accuracy, change: imp.accuracy_change, percent: true },
            { name: 'FPR', old: oldM.fpr, new: newM.fpr, change: imp.fpr_change, percent: true, inverse: true },
            { name: 'FNR', old: oldM.fnr, new: newM.fnr, change: imp.fnr_change, percent: true, inverse: true },
            { name: 'ROC-AUC', old: oldM.roc_auc, new: newM.roc_auc, change: imp.roc_auc_change }
        ];
        
        metrics.forEach((m, idx) => {
            const changeColor = m.inverse 
                ? (m.change < 0 ? 'text-green-400' : m.change > 0 ? 'text-red-400' : 'text-gray-400')
                : (m.change > 0 ? 'text-green-400' : m.change < 0 ? 'text-red-400' : 'text-gray-400');
            const changeIcon = m.change > 0 ? '↑' : m.change < 0 ? '↓' : '→';
            
            html += `
                <tr class="border-b border-slate-700">
                    ${idx === 0 ? `<td rowspan="4" class="font-bold align-top">${model}</td>` : ''}
                    <td>${m.name}</td>
                    <td>${m.old !== undefined ? (m.percent ? (m.old * 100).toFixed(2) + '%' : m.old.toFixed(4)) : '-'}</td>
                    <td>${m.new !== undefined ? (m.percent ? (m.new * 100).toFixed(2) + '%' : m.new.toFixed(4)) : '-'}</td>
                    <td class="${changeColor} font-bold">
                        ${changeIcon} ${m.change !== undefined ? Math.abs(m.change).toFixed(2) : '-'}${m.percent ? '%' : ''}
                    </td>
                </tr>
            `;
        });
    }
    
    html += '</tbody></table></div>';
    container.innerHTML = html;
}

function updateImprovementChart(improvements) {
    const ctx = document.getElementById('improvementChart')?.getContext('2d');
    if (!ctx || !improvements) return;
    
    if (improvementChart) improvementChart.destroy();
    
    const models = Object.keys(improvements);
    const fprChanges = models.map(m => -improvements[m].fpr_change || 0); // Negative because lower is better
    
    improvementChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{
                label: 'FPR Improvement (%)',
                data: fprChanges,
                backgroundColor: fprChanges.map(v => v >= 0 ? '#22c55e' : '#ef4444'),
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: { grid: { color: '#334155' } },
                y: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// =============================================================================
// ACTION FUNCTIONS
// =============================================================================

async function runEvaluation() {
    const checkboxes = document.querySelectorAll('.eval-model-cb:checked');
    const models = Array.from(checkboxes).map(cb => cb.value);
    
    if (models.length === 0) {
        showToast('Please select at least one model', 'warning');
        return;
    }
    
    try {
        document.getElementById('evalBtn').disabled = true;
        document.getElementById('evalBtn').innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Running...';
        
        const res = await fetch('/api/models/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ models })
        });
        
        const data = await res.json();
        showToast(data.message, data.status === 'ok' ? 'success' : 'error');
        
        // Start polling for completion
        startEvaluationPolling();
        
    } catch (err) {
        showToast('Error starting evaluation', 'error');
        document.getElementById('evalBtn').disabled = false;
        document.getElementById('evalBtn').innerHTML = '<i class="fas fa-play mr-2"></i>Start Evaluation';
    }
}

function startEvaluationPolling() {
    const poll = setInterval(async () => {
        try {
            const res = await fetch('/api/training/status');
            const data = await res.json();
            
            if (!data.training.running) {
                clearInterval(poll);
                document.getElementById('evalBtn').disabled = false;
                document.getElementById('evalBtn').innerHTML = '<i class="fas fa-play mr-2"></i>Start Evaluation';
                showToast('Evaluation completed!', 'success');
                await loadModelsData();
            }
        } catch (err) {
            clearInterval(poll);
        }
    }, 2000);
}

async function startTraining() {
    const checkboxes = document.querySelectorAll('.train-model-cb:checked');
    const models = Array.from(checkboxes).map(cb => cb.value);
    const epochs = parseInt(document.getElementById('trainEpochs').value);
    const batchSize = parseInt(document.getElementById('trainBatchSize').value);
    const lr = parseFloat(document.getElementById('trainLR').value);
    
    if (models.length === 0) {
        showToast('Please select at least one model', 'warning');
        return;
    }
    
    try {
        document.getElementById('trainStartBtn').disabled = true;
        document.getElementById('trainStopBtn').disabled = false;
        
        const res = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ models, epochs, batch_size: batchSize, learning_rate: lr })
        });
        
        const data = await res.json();
        showToast(data.message, data.status === 'ok' ? 'success' : 'error');
        
        if (data.status === 'ok') {
            document.getElementById('trainingProgressContainer').classList.remove('hidden');
            document.getElementById('trainingIdleMessage').classList.add('hidden');
            startTrainingPolling();
        }
        
    } catch (err) {
        showToast('Error starting training', 'error');
        document.getElementById('trainStartBtn').disabled = false;
        document.getElementById('trainStopBtn').disabled = true;
    }
}

async function stopTraining() {
    try {
        const res = await fetch('/api/training/stop', { method: 'POST' });
        const data = await res.json();
        showToast(data.message, data.status === 'ok' ? 'success' : 'error');
        
        if (trainingPollingInterval) {
            clearInterval(trainingPollingInterval);
        }
        
        document.getElementById('trainStartBtn').disabled = false;
        document.getElementById('trainStopBtn').disabled = true;
        
    } catch (err) {
        showToast('Error stopping training', 'error');
    }
}

function startTrainingPolling() {
    if (trainingPollingInterval) clearInterval(trainingPollingInterval);
    
    trainingPollingInterval = setInterval(async () => {
        try {
            const res = await fetch('/api/training/status');
            const data = await res.json();
            const t = data.training;
            
            document.getElementById('trainingCurrentModel').textContent = t.current_model || 'Initializing...';
            document.getElementById('trainingEpochInfo').textContent = `Epoch ${t.current_epoch}/${t.total_epochs}`;
            document.getElementById('trainingProgressBar').style.width = `${t.progress}%`;
            document.getElementById('trainingStatus').textContent = t.running ? 'Training in progress...' : 'Completed';
            
            // Update logs
            const logsContainer = document.getElementById('trainingLogs');
            if (t.logs && t.logs.length > 0) {
                logsContainer.innerHTML = t.logs.map(log => `<div class="text-gray-300">${log}</div>`).join('');
                logsContainer.scrollTop = logsContainer.scrollHeight;
            }
            
            if (!t.running) {
                clearInterval(trainingPollingInterval);
                document.getElementById('trainStartBtn').disabled = false;
                document.getElementById('trainStopBtn').disabled = true;
                showToast('Training completed!', 'success');
                await loadModelsData();
                await loadHistory();
            }
            
        } catch (err) {
            clearInterval(trainingPollingInterval);
        }
    }, 2000);
}

async function validateDatasetPath() {
    const path = document.getElementById('customDatasetPath').value.trim();
    if (!path) {
        showToast('Please enter a path', 'warning');
        return;
    }
    
    try {
        const res = await fetch('/api/dataset/set-path', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        
        const data = await res.json();
        const resultDiv = document.getElementById('datasetValidationResult');
        resultDiv.classList.remove('hidden');
        
        if (data.status === 'ok') {
            resultDiv.innerHTML = `
                <div class="bg-green-600/20 text-green-400 p-4 rounded-lg">
                    <i class="fas fa-check-circle mr-2"></i>${data.message}
                </div>
            `;
        } else {
            resultDiv.innerHTML = `
                <div class="bg-red-600/20 text-red-400 p-4 rounded-lg">
                    <i class="fas fa-exclamation-circle mr-2"></i>${data.message}
                </div>
            `;
        }
        
    } catch (err) {
        showToast('Error validating path', 'error');
    }
}

async function loadReport(type) {
    try {
        const res = await fetch(`/api/reports/${type}`);
        const data = await res.json();
        
        const container = document.getElementById('reportContent');
        if (data.status === 'ok') {
            container.innerHTML = `<pre class="text-gray-300 whitespace-pre-wrap">${data.report}</pre>`;
        } else {
            container.innerHTML = `<p class="text-red-400 text-center py-8">${data.message}</p>`;
        }
        
    } catch (err) {
        showToast('Error loading report', 'error');
    }
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear all history?')) return;
    
    try {
        await fetch('/api/history/clear', { method: 'POST' });
        showToast('History cleared', 'success');
        await loadHistory();
    } catch (err) {
        showToast('Error clearing history', 'error');
    }
}

// =============================================================================
// REAL-TIME MONITOR FUNCTIONS
// =============================================================================

function startReplay() {
    const speed = parseFloat(document.getElementById('speedSelect').value);
    
    fetch('/api/start_replay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ csv_file: 'demo_test.csv', speed })
    })
    .then(r => r.json())
    .then(data => {
        showToast('Replay started', 'success');
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('trafficLog').innerHTML = '';
    })
    .catch(err => showToast('Error starting replay', 'error'));
}

function stopReplay() {
    fetch('/api/stop_replay', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
        showToast('Replay stopped', 'success');
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    })
    .catch(err => showToast('Error stopping replay', 'error'));
}

function updateLiveMonitor(data) {
    const { packet_id, true_label, predictions, stats } = data;
    
    document.getElementById('packetCount').textContent = packet_id;
    
    // Update live cards dynamically for all models
    Object.keys(predictions).forEach(modelName => {
        const pred = predictions[modelName];
        if (!pred) return;
        
        const nameLower = modelName.toLowerCase();
        const prefix = 'live' + modelName.charAt(0).toUpperCase() + modelName.slice(1);
        
        // Try to update elements (they may not exist for dynamically added models)
        const predEl = document.getElementById(`${prefix}Pred`);
        const confEl = document.getElementById(`${prefix}Conf`);
        const progressEl = document.getElementById(`${prefix}Progress`);
        const statusEl = document.getElementById(`${prefix}Status`);
        const correctEl = document.getElementById(`${prefix}Correct`);
        const wrongEl = document.getElementById(`${prefix}Wrong`);
        const card = document.getElementById(`liveCard${modelName}`);
        
        if (predEl) predEl.textContent = pred.pred === 1 ? '🔴 Attack' : '🟢 Normal';
        if (confEl) confEl.textContent = (pred.confidence * 100).toFixed(1) + '%';
        if (progressEl) progressEl.style.width = (pred.prob * 100) + '%';
        if (statusEl) statusEl.textContent = pred.pred === 1 ? '🔴' : '🟢';
        
        const correct = stats[`${nameLower}_correct`] || 0;
        const wrong = stats[`${nameLower}_wrong`] || 0;
        if (correctEl) correctEl.textContent = correct;
        if (wrongEl) wrongEl.textContent = wrong;
        
        // Add border effect
        if (card && pred.pred === 1) {
            card.classList.add('border-red-500');
            setTimeout(() => card.classList.remove('border-red-500'), 500);
        }
    });
    
    // Update live chart dynamically for all models
    if (liveChart && liveChart.data.datasets.length > 0) {
        liveChart.data.labels.push(packet_id);
        
        // Update each dataset based on model name
        liveChart.data.datasets.forEach((dataset) => {
            const modelName = dataset.label;
            const prob = predictions[modelName]?.prob || 0;
            dataset.data.push(prob);
        });
        
        if (liveChart.data.labels.length > 50) {
            liveChart.data.labels.shift();
            liveChart.data.datasets.forEach(ds => ds.data.shift());
        }
        
        liveChart.update('none');
    }
    
    // Add to log - dynamically build model predictions
    const log = document.getElementById('trafficLog');
    const trueLbl = true_label === 1 ? '🔴 Attack' : '🟢 Normal';
    const modelPreds = Object.keys(predictions).map(m => 
        `<span class="ml-2">${m}: ${predictions[m]?.pred === 1 ? '🔴' : '🟢'}</span>`
    ).join('');
    
    const entry = document.createElement('div');
    entry.className = 'py-1 border-b border-slate-800';
    entry.innerHTML = `
        <span class="text-gray-500">#${packet_id}</span>
        <span class="ml-4">True: ${trueLbl}</span>
        ${modelPreds}
    `;
    log.insertBefore(entry, log.firstChild);
    while (log.children.length > 100) log.removeChild(log.lastChild);
}

// =============================================================================
// UI HELPER FUNCTIONS
// =============================================================================

function showTab(tabName) {
    // Update sidebar
    document.querySelectorAll('.sidebar-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.tab === tabName) item.classList.add('active');
    });
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.getElementById(`tab-${tabName}`).classList.add('active');
    
    // Update header
    const titles = {
        dashboard: ['Dashboard', 'Tổng quan hệ thống phát hiện DDoS'],
        models: ['Model Evaluation', 'Đánh giá và so sánh các models'],
        realtime: ['Real-time Monitor', 'Giám sát traffic thời gian thực'],
        training: ['Training', 'Quản lý training models'],
        dataset: ['Dataset Manager', 'Quản lý dữ liệu training'],
        compare: ['Compare Results', 'So sánh kết quả cũ và mới'],
        history: ['History & Reports', 'Lịch sử training và báo cáo']
    };
    
    document.getElementById('pageTitle').textContent = titles[tabName]?.[0] || tabName;
    document.getElementById('pageSubtitle').textContent = titles[tabName]?.[1] || '';
    
    currentTab = tabName;
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const icon = document.getElementById('toastIcon');
    const msg = document.getElementById('toastMessage');
    
    const icons = {
        info: 'fa-info-circle text-blue-400',
        success: 'fa-check-circle text-green-400',
        warning: 'fa-exclamation-circle text-yellow-400',
        error: 'fa-times-circle text-red-400'
    };
    
    icon.className = `fas ${icons[type] || icons.info}`;
    msg.textContent = message;
    
    toast.classList.remove('translate-y-20', 'opacity-0');
    
    setTimeout(() => {
        toast.classList.add('translate-y-20', 'opacity-0');
    }, 3000);
}

function updateClock() {
    const now = new Date();
    document.getElementById('currentTime').textContent = now.toLocaleString('vi-VN');
}

async function refreshData() {
    showToast('Refreshing data...', 'info');
    await loadInitialData();
    showToast('Data refreshed', 'success');
}
