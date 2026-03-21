/**
 * Frontend JavaScript for Student Stress Predictor Dashboard
 */

const API_URL = "http://localhost:8000";

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    checkAPIStatus();
    setupFormHandler();
});

/**
 * Check API health status
 */
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            document.getElementById("apiStatus").textContent = "✅ Connected";
        } else {
            document.getElementById("apiStatus").textContent = "⚠️ API Error";
        }
    } catch (error) {
        document.getElementById("apiStatus").textContent = "❌ Offline";
        console.error("API connection failed:", error);
    }
}

/**
 * Setup form submission handler
 */
function setupFormHandler() {
    const form = document.getElementById("predictorForm");
    if (form) {
        form.addEventListener("submit", handleFormSubmit);
    }
}

/**
 * Handle form submission for prediction
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    
    // Build StudentInput object
    const studentData = {
        attendance_pct: parseFloat(formData.get("attendance_pct")),
        study_hours: parseInt(formData.get("study_hours")),
        sleep_hrs: parseFloat(formData.get("sleep_hrs")),
        screen_time: parseFloat(formData.get("screen_time")),
        activity_days: parseInt(formData.get("activity_days")),
        extracurricular: parseInt(formData.get("extracurricular")),
        part_time_work: formData.get("part_time_work") ? 1 : 0,
        accommodation: parseInt(formData.get("accommodation")),
        phone_midnight: parseInt(formData.get("phone_midnight")),
        stress_level: parseInt(formData.get("stress_level")),
        overwhelmed: parseInt(formData.get("overwhelmed")),
        social_support: parseInt(formData.get("social_support")),
        skips_meals: parseInt(formData.get("skips_meals")),
        life_satisfaction: parseInt(formData.get("life_satisfaction"))
    };
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(studentData)
        });
        
        if (response.ok) {
            const result = await response.json();
            displayPredictionResult(result);
        } else {
            alert("Error getting prediction. Please check the API.");
        }
    } catch (error) {
        console.error("Prediction error:", error);
        alert("Failed to get prediction. Is the API running?");
    }
}

/**
 * Display prediction result
 */
function displayPredictionResult(result) {
    const resultDiv = document.getElementById("predictionResult");
    const resultCard = document.getElementById("resultCard");
    
    const statusColor = result.label === 1 ? "#27ae60" : "#e74c3c";
    const statusEmoji = result.label === 1 ? "✅" : "⚠️";
    
    const clusterDescriptions = {
        0: "You have healthy lifestyle habits with good work-life balance.",
        1: "You're working hard but need more sleep and stress management.",
        2: "Your stress levels are high. Consider taking breaks and seeking support."
    };
    
    resultCard.innerHTML = `
        <div style="background-color: ${statusColor}; padding: 20px; border-radius: 8px; color: white; margin-bottom: 20px;">
            <h2>${statusEmoji} ${result.label_text}</h2>
            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
        </div>
        
        <div style="background-color: #3498db; padding: 20px; border-radius: 8px; color: white;">
            <h3>📊 Lifestyle Archetype</h3>
            <p><strong>${result.cluster_name}</strong></p>
            <p>${clusterDescriptions[result.cluster_id] || "View your cluster profile."}</p>
        </div>
    `;
    
    resultDiv.style.display = "block";
}

/**
 * Update range input display values
 */
function updateValue(elementId, value) {
    document.getElementById(elementId).textContent = value;
}

/**
 * Switch between tabs
 */
function showTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll(".tab-content");
    tabs.forEach(tab => tab.classList.remove("active"));
    
    // Remove active class from all buttons
    const buttons = document.querySelectorAll(".tab-button");
    buttons.forEach(btn => btn.classList.remove("active"));
    
    // Show selected tab
    document.getElementById(tabName).classList.add("active");
    
    // Add active class to clicked button
    event.target.classList.add("active");
}

/**
 * Load metrics from API (to be called when models are ready)
 */
async function loadMetrics() {
    try {
        const response = await fetch(`${API_URL}/metrics`);
        if (response.ok) {
            const metrics = await response.json();
            displayMetricsChart(metrics);
        }
    } catch (error) {
        console.error("Error loading metrics:", error);
    }
}

/**
 * Display metrics comparison chart
 */
function displayMetricsChart(metrics) {
    const ctx = document.getElementById("metricsChart");
    if (!ctx) return;
    
    // Chart.js chart setup would go here
    console.log("Metrics loaded:", metrics);
}

/**
 * Load clusters from API (to be called when clustering is ready)
 */
async function loadClusters() {
    try {
        const response = await fetch(`${API_URL}/clusters`);
        if (response.ok) {
            const clusters = await response.json();
            displayClusters(clusters);
        }
    } catch (error) {
        console.error("Error loading clusters:", error);
    }
}

/**
 * Display cluster information
 */
function displayClusters(clustersData) {
    console.log("Clusters loaded:", clustersData);
    // Cluster display logic would go here
}
