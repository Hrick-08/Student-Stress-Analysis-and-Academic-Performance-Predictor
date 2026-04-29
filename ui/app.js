/**
 * Student Stress Predictor Dashboard - JavaScript
 * Handles tab navigation, model comparison, and live prediction
 */

// const API_BASE = "https://student-stress-analysis-and-academic.onrender.com";
const API_BASE = "http://localhost:8001";

// Safely joins API_BASE + path, preventing double slashes
function apiUrl(path) {
    return `${API_BASE.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}
let chartsInitialized = false;
let metricsData = {};
let selectedModel = "AdaBoost";

// Initialize on page load
document.addEventListener("DOMContentLoaded", function () {
    console.log("🚀 Student Stress Predictor Dashboard Loaded");
    console.log(`🔗 API Base URL: ${API_BASE}`);
    loadMetrics();
});

/**
 * Show/hide tabs
 */
function showTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll(".tab-content");
    tabs.forEach(tab => tab.classList.remove("active"));

    // Show selected tab
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add("active");
    }

    // Update button styling
    const buttons = document.querySelectorAll(".tab-button");
    buttons.forEach(btn => btn.classList.remove("active"));
    event.target.classList.add("active");
}

/**
 * Load model metrics from API
 */
async function loadMetrics() {
    try {
        console.log("📥 Loading model metrics from API...");
        const response = await fetch(apiUrl("models"));
        const data = await response.json();
        console.log("✅ Metrics loaded:", data.models);

        metricsData = {};
        data.models.forEach(model => {
            metricsData[model.name] = {
                r2: model.test_r2,
                rmse: model.rmse
            };
        });

        populateMetricsTable(data.models);
        
        // Initialize charts immediately after loading metrics
        if (!chartsInitialized) {
            initializeCharts();
            chartsInitialized = true;
        }
    } catch (error) {
        console.error("❌ Error loading metrics:", error);
    }
}

/**
 * Populate metrics table
 */
function populateMetricsTable(models) {
    const tbody = document.getElementById("metricsTableBody");
    tbody.innerHTML = "";

    models.forEach(model => {
        const row = document.createElement("tr");
        const status = model.name === "AdaBoost" ? "⭐ Recommended" : "Available";
        const statusClass = model.name === "AdaBoost" ? "status-best" : "status-normal";

        row.innerHTML = `
            <td><strong>${model.name}</strong></td>
            <td>${model.test_r2.toFixed(4)}</td>
            <td>${model.rmse.toFixed(4)}</td>
            <td><span class="${statusClass}">${status}</span></td>
        `;
        tbody.appendChild(row);
    });
}

/**
 * Initialize Chart.js charts for metrics
 */
let r2Chart = null;
let rmseChart = null;

function initializeCharts() {
    if (!metricsData || Object.keys(metricsData).length === 0) {
        console.error("No metrics data available");
        return;
    }

    const modelNames = Object.keys(metricsData).sort((a, b) => 
        metricsData[b].r2 - metricsData[a].r2
    );
    const r2Values = modelNames.map(name => metricsData[name].r2);
    const rmseValues = modelNames.map(name => metricsData[name].rmse);

    // Destroy existing charts
    if (r2Chart) r2Chart.destroy();
    if (rmseChart) rmseChart.destroy();

    // R² Chart
    const r2Ctx = document.getElementById("r2Chart");
    if (r2Ctx) {
        r2Chart = new Chart(r2Ctx, {
            type: "bar",
            data: {
                labels: modelNames,
                datasets: [{
                    label: "Test R²",
                    data: r2Values,
                    backgroundColor: [
                        "#2ecc71", // Green for best
                        "#3498db", "#3498db", "#3498db", "#3498db", "#e74c3c"
                    ],
                    borderColor: "#2c3e50",
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y.toFixed(4);
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: { callback: value => value.toFixed(2) }
                    }
                }
            }
        });
    }

    // RMSE Chart
    const rmseCtx = document.getElementById("rmseChart");
    if (rmseCtx) {
        rmseChart = new Chart(rmseCtx, {
            type: "bar",
            data: {
                labels: modelNames,
                datasets: [{
                    label: "RMSE (lower is better)",
                    data: rmseValues,
                    backgroundColor: [
                        "#2ecc71",
                        "#3498db", "#3498db", "#3498db", "#3498db", "#e74c3c"
                    ],
                    borderColor: "#2c3e50",
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y.toFixed(4);
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { callback: value => value.toFixed(2) }
                    }
                }
            }
        });
    }
}

/**
 * Update range slider display value
 */
function updateValue(fieldId) {
    const input = document.getElementById(fieldId);
    const display = document.getElementById(fieldId + "-val");
    if (display) {
        display.textContent = input.value;
    }
}

/**
 * Update selected model
 */
function updateSelectedModel() {
    const select = document.getElementById("modelSelect");
    selectedModel = select.value;
    
    const note = document.getElementById("modelNote");
    const metricsEntry = metricsData[selectedModel];
    if (metricsEntry && note) {
        note.textContent = `Using ${selectedModel} (R² = ${metricsEntry.r2.toFixed(4)})`;
    }
}

/**
 * Collect form data and run prediction
 */
async function runPrediction() {
    // Collect all input values
    const payload = {
        safety: parseInt(document.getElementById("safety").value),
        basic_needs: parseInt(document.getElementById("basic_needs").value),
        living_conditions: parseInt(document.getElementById("living_conditions").value),
        anxiety_level: parseInt(document.getElementById("anxiety_level").value),
        depression: parseInt(document.getElementById("depression").value),
        self_esteem: parseInt(document.getElementById("self_esteem").value),
        peer_pressure: parseInt(document.getElementById("peer_pressure").value),
        study_load: parseInt(document.getElementById("study_load").value),
        future_career_concerns: parseInt(document.getElementById("future_career_concerns").value),
        teacher_student_relationship: parseInt(document.getElementById("teacher_student_relationship").value),
        social_support: parseInt(document.getElementById("social_support").value),
        mental_health_history: parseInt(document.getElementById("mental_health_history").value),
        headache: parseInt(document.getElementById("headache").value),
        blood_pressure: parseInt(document.getElementById("blood_pressure").value),
        sleep_quality: parseInt(document.getElementById("sleep_quality").value),
        breathing_problem: parseInt(document.getElementById("breathing_problem").value),
        extracurricular_activities: parseInt(document.getElementById("extracurricular_activities").value),
        stress_level: parseInt(document.getElementById("stress_level").value)
    };

    console.log("📤 Sending prediction request with payload:", payload);
    console.log(`🔗 API Endpoint: ${apiUrl(`predict?model_name=${selectedModel}`)}`)

    try {
        // Get prediction from selected model
        const response = await fetch(
            apiUrl(`predict?model_name=${selectedModel}`),
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            }
        );

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();
        console.log("✅ Prediction received from API:", result);
        displayResult(result, payload);

        // Also get all models predictions for comparison
        getAllModelsPredictions(payload);

    } catch (error) {
        console.error("❌ Error:", error);
        displayError(error.message);
    }
}

/**
 * Display prediction result
 */
function displayResult(result, inputData) {
    const container = document.getElementById("resultContainer");
    
    // Convert 0-5 scale to percentage (0-100%)
    // Note: The model predicts on 0-5 scale, we convert to percentage
    const percentage = Math.round((result.predicted_score / 5) * 100);
    
    console.log("📊 Displaying result:", {
        rawScore: result.predicted_score,
        percentage: percentage,
        band: result.performance_band,
        model: result.model_name,
        r2: result.model_r2,
        rmse: result.model_rmse
    });
    
    const bandClass = getBandClass(result.performance_band);
    
    container.innerHTML = `
        <div class="result-card ${bandClass}">
            <div class="result-header">
                <h3>Academic Performance Score</h3>
                <p class="result-percentage">${percentage}%</p>
                <p class="result-scale">on 0-100% scale</p>
            </div>
            
            <div class="result-band">
                <p><strong>Status:</strong></p>
                <div class="band-badge ${bandClass}">${result.performance_band}</div>
            </div>
            
            <div class="result-details">
                <p><strong>Raw Model Prediction:</strong> ${result.predicted_score.toFixed(2)} / 5.0</p>
                <p><strong>Model Used:</strong> ${result.model_name}</p>
                <p><strong>Model Test R²:</strong> ${result.model_r2.toFixed(4)}</p>
                <p><strong>Model RMSE:</strong> ${result.model_rmse.toFixed(4)}</p>
            </div>
            
            <div class="result-interpretation">
                <p><strong>Interpretation:</strong></p>
                <p>${getInterpretation(result.performance_band, result.predicted_score, inputData)}</p>
            </div>
        </div>
    `;
}

/**
 * Get all models predictions
 */
async function getAllModelsPredictions(payload) {
    try {
        console.log("🌟 Requesting all models predictions...");
        const response = await fetch(
            apiUrl("predict-all"),
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            }
        );

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();
        console.log("✅ All models predictions received:", result.predictions);
        displayAllModelsPredictions(result.predictions);

    } catch (error) {
        console.error("❌ Error loading all models:", error);
    }
}

/**
 * Display all models predictions table
 */
function displayAllModelsPredictions(predictions) {
    const container = document.getElementById("allModelsContainer");
    const tbody = document.getElementById("allModelsBody");
    
    console.log("📋 Building comparison table for all models");

    tbody.innerHTML = "";
    
    // Convert to array and sort by risk level
    const predictionArray = Object.entries(predictions)
        .filter(([name, pred]) => pred !== null)
        .map(([name, pred]) => ({ name, pred }));
    
    // Define risk order (At Risk first, then Average, then Performing)
    const riskOrder = { "At Risk": 0, "Average": 1, "Performing": 2 };
    
    // Sort by band risk level
    predictionArray.sort((a, b) => {
        const riskA = riskOrder[a.pred.performance_band] || 999;
        const riskB = riskOrder[b.pred.performance_band] || 999;
        return riskA - riskB;
    });

    predictionArray.forEach(({ name, pred }) => {
        const percentage = Math.round((pred.predicted_score / 5) * 100);
        const bandClass = getBandClass(pred.performance_band);
        console.log(`  • ${name}: ${percentage}% (${pred.performance_band}) - Raw: ${pred.predicted_score.toFixed(2)}`);
        
        const row = document.createElement("tr");
        row.innerHTML = `
            <td><strong>${name}</strong></td>
            <td>${percentage}%</td>
            <td><span class="band-badge ${bandClass}">${pred.performance_band}</span></td>
            <td>${pred.predicted_score.toFixed(2)} / 5</td>
            <td>${pred.model_r2.toFixed(4)}</td>
        `;
        tbody.appendChild(row);
    });

    container.style.display = "block";
}

/**
 * Get CSS class for performance band
 */
function getBandClass(band) {
    if (band === "At Risk") return "at-risk";
    if (band === "Average") return "average";
    return "performing";
}

/**
 * Get color for performance band
 */
function getBandColor(band) {
    if (band === "At Risk") return "#e74c3c";
    if (band === "Average") return "#f39c12";
    return "#2ecc71";
}

/**
 * Get interpretation text based on prediction
 */
function getInterpretation(band, score, inputData) {
    let interpretation = "";
    
    if (band === "At Risk") {
        interpretation = `⚠️ <strong>${score.toFixed(0)} - At Risk:</strong> This student may struggle academically. `;
        interpretation += "Key areas of concern: ";
        
        const concerns = [];
        if (inputData.stress_level === 2) concerns.push("High stress level");
        if (inputData.anxiety_level > 15) concerns.push("High anxiety");
        if (inputData.sleep_quality === 0) concerns.push("Poor sleep quality");
        if (inputData.self_esteem < 10) concerns.push("Low self-esteem");
        if (inputData.study_load > 3) concerns.push("High study load");
        
        interpretation += concerns.join(", ") || "Multiple lifestyle factors";
        interpretation += ". Recommend support and intervention.";
    } else if (band === "Average") {
        interpretation = `📊 <strong>${score.toFixed(0)} - Average:</strong> This student's performance is moderate. `;
        interpretation += "There's room for improvement in: ";
        
        const improvements = [];
        if (inputData.stress_level > 0) improvements.push("stress management");
        if (inputData.sleep_quality < 2) improvements.push("sleep quality");
        if (inputData.social_support < 3) improvements.push("social support");
        if (inputData.study_load > 2) improvements.push("work-life balance");
        
        interpretation += improvements.join(", ") || "lifestyle habits";
        interpretation += ". Small changes can lead to better results.";
    } else {
        interpretation = `✅ <strong>${score.toFixed(0)} - Performing:</strong> This student is performing well! `;
        interpretation += "Positive factors include: ";
        
        const positives = [];
        if (inputData.stress_level === 0) positives.push("low stress");
        if (inputData.sleep_quality === 2) positives.push("good sleep");
        if (inputData.social_support > 3) positives.push("strong support network");
        if (inputData.self_esteem > 20) positives.push("high self-esteem");
        
        interpretation += positives.join(", ") || "good lifestyle habits";
        interpretation += ". Continue current practices.";
    }
    
    return interpretation;
}

/**
 * Display error message
 */
function displayError(message) {
    const container = document.getElementById("resultContainer");
    container.innerHTML = `
        <div class="error-box">
            <p>❌ Error: ${message}</p>
            <p>Please check your inputs and try again.</p>
        </div>
    `;
}