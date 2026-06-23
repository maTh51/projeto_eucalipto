// =====================================================================
// GLOBAL STATE & API CONFIG
// =====================================================================
const API_BASE = window.location.origin;

let currentLoadedFile = null;
let currentMetadata = null;
let currentTreeData = null;
let activePointCloudObj = null;
let activeMeshObj = null;
let batchResults = [];

// Three.js instances
let scene, camera, renderer, controls;
let isThreeJSActive = false;

// Method Parameters Configurations
const dbhMethods = {
    ensemble: [
        { name: 'breast_height_offset', label: 'Breast Height Offset (m)', min: 0.5, max: 2.5, val: 1.3, step: 0.1 },
        { name: 'offset_range', label: 'Offset Range (m)', min: 0.1, max: 0.8, val: 0.3, step: 0.05 },
        { name: 'n_slices', label: 'Slices Count', min: 3, max: 15, val: 7, step: 1 },
        { name: 'slice_thickness', label: 'Slice Thickness (m)', min: 0.05, max: 0.5, val: 0.16, step: 0.02 },
        { name: 'ransac_thresh', label: 'RANSAC Thresh (m)', min: 0.005, max: 0.1, val: 0.02, step: 0.005 }
    ],
    single_ransac: [
        { name: 'breast_height_offset', label: 'Breast Height Offset (m)', min: 0.5, max: 2.5, val: 1.3, step: 0.1 },
        { name: 'slice_thickness', label: 'Slice Thickness (m)', min: 0.05, max: 0.5, val: 0.16, step: 0.02 },
        { name: 'ransac_thresh', label: 'RANSAC Thresh (m)', min: 0.005, max: 0.1, val: 0.02, step: 0.005 }
    ],
    ls: [
        { name: 'breast_height_offset', label: 'Breast Height Offset (m)', min: 0.5, max: 2.5, val: 1.3, step: 0.1 },
        { name: 'slice_thickness', label: 'Slice Thickness (m)', min: 0.05, max: 0.5, val: 0.16, step: 0.02 },
        { name: 'radius_min', label: 'Min Radius (m)', min: 0.01, max: 0.5, val: 0.01, step: 0.01 },
        { name: 'radius_max', label: 'Max Radius (m)', min: 0.2, max: 2.0, val: 1.0, step: 0.05 }
    ]
};

const volumeMethods = {
    cylinder: [], 
    voxel: [
        { name: 'voxel_size', label: 'Voxel Size (m)', min: 0.01, max: 0.2, val: 0.05, step: 0.01 }
    ],
    axis_profile: [
        { name: 'n_slices', label: 'Slices Count', min: 5, max: 50, val: 20, step: 1 },
        { name: 'slice_thickness', label: 'Slice Thickness (m)', min: 0.05, max: 1.0, val: 0.30, step: 0.05 },
        { name: 'radius_percentile', label: 'Radius Percentile (%)', min: 50, max: 99, val: 85, step: 1 },
        { name: 'min_points_per_slice', label: 'Min Points/Slice', min: 3, max: 100, val: 20, step: 1 }
    ],
    frustum: [
        { name: 'n_height_samples', label: 'Taper Samples', min: 5, max: 30, val: 15, step: 1 },
        { name: 'slice_thickness', label: 'Slice Thickness (m)', min: 0.05, max: 0.5, val: 0.20, step: 0.02 },
        { name: 'ransac_thresh', label: 'RANSAC Thresh (m)', min: 0.005, max: 0.1, val: 0.03, step: 0.005 },
        { name: 'radius_min', label: 'Min Radius (m)', min: 0.01, max: 0.3, val: 0.01, step: 0.01 },
        { name: 'radius_max', label: 'Max Radius (m)', min: 0.3, max: 2.0, val: 1.0, step: 0.05 },
        { name: 'min_inliers', label: 'Min Inliers/Slice', min: 3, max: 20, val: 3, step: 1 }
    ],
    taper: [
        { name: 'n_height_samples', label: 'Taper Samples', min: 5, max: 30, val: 15, step: 1 },
        { name: 'slice_thickness', label: 'Slice Thickness (m)', min: 0.05, max: 0.5, val: 0.20, step: 0.02 },
        { name: 'ransac_thresh', label: 'RANSAC Thresh (m)', min: 0.005, max: 0.1, val: 0.03, step: 0.005 },
        { name: 'radius_min', label: 'Min Radius (m)', min: 0.01, max: 0.3, val: 0.01, step: 0.01 },
        { name: 'radius_max', label: 'Max Radius (m)', min: 0.3, max: 2.0, val: 1.0, step: 0.05 },
        { name: 'min_inliers', label: 'Min Inliers/Slice', min: 3, max: 20, val: 3, step: 1 }
    ]
};

// =====================================================================
// INITIALIZATION
// =====================================================================
document.addEventListener("DOMContentLoaded", () => {
    initUI();
    initThreeJS();
    fetchFileList();
});

// =====================================================================
// UI LOGIC & EVENTS
// =====================================================================
function initUI() {
    // Buttons
    document.getElementById("btn-load-file").addEventListener("click", handleLoadFile);
    document.getElementById("btn-calculate").addEventListener("click", handleCalculate);
    document.getElementById("btn-reset-cam").addEventListener("click", resetCamera);
    document.getElementById("btn-show-plot").addEventListener("click", handleShowPlot);
    document.getElementById("color-mode-select").addEventListener("change", handleColorModeChange);
    
    // File upload elements
    const fileUploadBtn = document.getElementById("btn-upload-file");
    const fileUploadInput = document.getElementById("file-upload-input");
    
    fileUploadBtn.addEventListener("click", () => fileUploadInput.click());
    fileUploadInput.addEventListener("change", handleFileUpload);
    
    // Batch process elements
    document.getElementById("btn-run-batch").addEventListener("click", handleRunBatch);
    document.getElementById("table-search").addEventListener("input", handleTableSearch);
    
    // Selects
    document.getElementById("tree-select").addEventListener("change", handleTreeChange);
    document.getElementById("dbh-method").addEventListener("change", renderDBHSliders);
    document.getElementById("volume-method").addEventListener("change", renderVolumeSliders);
    
    // Tab Triggers
    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            if (btn.disabled) return;
            
            // Switch tabs active button
            tabButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            // Switch tab content visibility
            const targetId = btn.getAttribute("data-tab");
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            document.getElementById(targetId).classList.add("active");
            
            // Redraw/reset camera when switching view kinds
            if (targetId === "macro-view-tab" && batchResults.length > 0) {
                renderPlotMeshes(batchResults);
            }
        });
    });

    // Toolbar toggle controls
    const togglePointsBtn = document.getElementById("toggle-points");
    togglePointsBtn.addEventListener("click", () => {
        togglePointsBtn.classList.toggle("active");
        if (activePointCloudObj) {
            activePointCloudObj.visible = togglePointsBtn.classList.contains("active");
        }
    });

    const toggleMeshBtn = document.getElementById("toggle-mesh");
    toggleMeshBtn.addEventListener("click", () => {
        toggleMeshBtn.classList.toggle("active");
        if (activeMeshObj) {
            activeMeshObj.visible = toggleMeshBtn.classList.contains("active");
        }
    });

    const toggleLabelsBtn = document.getElementById("toggle-labels");
    toggleLabelsBtn.addEventListener("click", () => {
        toggleLabelsBtn.classList.toggle("active");
        if (typeof updateHTMLLabels === "function") {
            updateHTMLLabels();
        }
    });

    // Toolbar sliders
    document.getElementById("point-size-slider").addEventListener("input", (e) => {
        if (activePointCloudObj && activePointCloudObj.material) {
            activePointCloudObj.material.size = parseFloat(e.target.value) / 100;
        }
    });

    document.getElementById("mesh-opacity-slider").addEventListener("input", (e) => {
        if (activeMeshObj && activeMeshObj.material) {
            activeMeshObj.material.opacity = parseFloat(e.target.value) / 100;
        }
    });
}

// Generate controls HTML
function renderDBHSliders() {
    const method = document.getElementById("dbh-method").value;
    const container = document.getElementById("dbh-params-container");
    container.innerHTML = "";
    
    const params = dbhMethods[method] || [];
    params.forEach(p => {
        container.appendChild(createSliderElement("dbh", p.name, p.label, p.min, p.max, p.val, p.step));
    });
}

function renderVolumeSliders() {
    const method = document.getElementById("volume-method").value;
    const container = document.getElementById("volume-params-container");
    container.innerHTML = "";
    
    const params = volumeMethods[method] || [];
    params.forEach(p => {
        container.appendChild(createSliderElement("vol", p.name, p.label, p.min, p.max, p.val, p.step));
    });
}

function createSliderElement(prefix, name, label, min, max, val, step) {
    const id = `${prefix}-param-${name}`;
    
    const group = document.createElement("div");
    group.className = "slider-group";
    
    const header = document.createElement("div");
    header.className = "slider-header";
    header.innerHTML = `<span>${label}</span><span class="slider-val" id="${id}-val">${val}</span>`;
    
    const slider = document.createElement("input");
    slider.type = "range";
    slider.id = id;
    slider.min = min;
    slider.max = max;
    slider.value = val;
    slider.step = step;
    
    slider.addEventListener("input", (e) => {
        document.getElementById(`${id}-val`).innerText = e.target.value;
    });
    
    group.appendChild(header);
    group.appendChild(slider);
    return group;
}

// Read parameters from UI
function getParameters(prefix, paramsList) {
    const params = {};
    paramsList.forEach(p => {
        const el = document.getElementById(`${prefix}-param-${p.name}`);
        if (el) {
            params[p.name] = parseFloat(el.value);
        }
    });
    return params;
}

// =====================================================================
// API CALLS & LOAD HANDLERS
// =====================================================================
async function fetchFileList() {
    try {
        const res = await fetch(`${API_BASE}/api/files`);
        if (!res.ok) throw new Error("Could not retrieve file list.");
        const files = await res.json();
        
        const select = document.getElementById("file-select");
        select.innerHTML = '<option value="" disabled selected>Select point cloud file...</option>';
        files.forEach(f => {
            select.innerHTML += `<option value="${f.path}">${f.path} (${f.size_mb} MB)</option>`;
        });
        printLog("Loaded workspaces and available point clouds list.", "success");
    } catch (err) {
        printLog(`Error: ${err.message}`, "error");
    }
}

async function handleLoadFile() {
    const fileSelect = document.getElementById("file-select");
    const filepath = fileSelect.value;
    if (!filepath) {
        alert("Please select a file first.");
        return;
    }
    
    showLoading("Reading point cloud metadata...");
    try {
        const res = await fetch(`${API_BASE}/api/load`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filepath })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Load failed.");
        
        handleLoadedMetadata(data);
    } catch (err) {
        printLog(`Load Error: ${err.message}`, "error");
        alert(`Failed to load file: ${err.message}`);
    } finally {
        hideLoading();
    }
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append("file", file);
    
    showLoading(`Uploading ${file.name} to server...`);
    try {
        const res = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Upload failed.");
        
        await fetchFileList();
        handleLoadedMetadata(data);
        document.getElementById("file-select").value = data.filepath;
    } catch (err) {
        printLog(`Upload Error: ${err.message}`, "error");
        alert(`Failed to upload file: ${err.message}`);
    } finally {
        document.getElementById("file-upload-input").value = "";
        hideLoading();
    }
}

function handleLoadedMetadata(data) {
    currentLoadedFile = data.filepath;
    currentMetadata = data;
    
    // Update header status
    const statusDot = document.querySelector(".status-dot");
    statusDot.className = "status-dot status-online";
    document.querySelector("#status-display .status-text").innerText = `Loaded: ${data.filename || PathBasename(data.filepath)}`;
    
    // Populate tree IDs
    const treeSelect = document.getElementById("tree-select");
    treeSelect.innerHTML = '<option value="" disabled selected>Select a Tree Segment ID...</option>';
    data.tree_ids.forEach(id => {
        treeSelect.innerHTML += `<option value="${id}">Tree Segment #${id}</option>`;
    });
    
    treeSelect.disabled = false;
    document.getElementById("btn-show-plot").disabled = false;
    document.getElementById("dbh-method").disabled = false;
    document.getElementById("volume-method").disabled = false;
    document.getElementById("wood-density").disabled = false;
    document.getElementById("tab-btn-macro").disabled = false;
    
    renderDBHSliders();
    renderVolumeSliders();
    
    // Reset batch view state
    batchResults = [];
    document.getElementById("macro-summary-section").style.display = "none";
    document.getElementById("macro-table-section").style.display = "none";
    
    const errBanner = document.getElementById("single-tree-error-banner");
    if (errBanner) errBanner.style.display = "none";
    
    // Clear viewport and render the plot point cloud
    clearMesh();
    renderPlotPointCloud(data.plot_points, data.plot_colors, data.plot_is_trunk);
    
    printLog(`Successfully loaded point cloud: ${data.filepath}`, "success");
    printLog(`Total Points: ${data.num_points.toLocaleString()}`, "log");
    printLog(`Unique Trees Segmented: ${data.tree_count}`, "highlight");
    printLog(`Tree ID Column: '${data.tree_id_column}' | Semantic label: '${data.trunk_column}'`, "log");
}

async function handleTreeChange() {
    const treeId = document.getElementById("tree-select").value;
    if (!treeId) return;
    
    showLoading("Retrieving tree points...");
    try {
        const res = await fetch(`${API_BASE}/api/tree/${treeId}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to load tree points.");
        
        currentTreeData = data;
        clearMesh();
        
        // Render single tree point cloud
        renderPointCloud(data.points, data.colors, data.is_trunk);
        
        document.getElementById("btn-calculate").disabled = false;
        clearMetricsDisplay();
        
        const errBanner = document.getElementById("single-tree-error-banner");
        if (errBanner) errBanner.style.display = "none";
        
        printLog(`Fetched Tree ID: #${treeId}`, "success");
        printLog(`Points: ${data.num_points.toLocaleString()} | Height: ${data.bounds.height.toFixed(2)}m`, "log");
        
        // Trigger quick fit automatically
        handleCalculate();
    } catch (err) {
        printLog(`Tree load error: ${err.message}`, "error");
        alert(err.message);
    } finally {
        hideLoading();
    }
}

async function handleCalculate() {
    if (!currentTreeData) return;
    
    const treeId = document.getElementById("tree-select").value;
    const dbhMethod = document.getElementById("dbh-method").value;
    const volumeMethod = document.getElementById("volume-method").value;
    const woodDensity = parseFloat(document.getElementById("wood-density").value) || 900.0;
    
    const dbhParams = getParameters("dbh", dbhMethods[dbhMethod] || []);
    const volumeParams = getParameters("vol", volumeMethods[volumeMethod] || []);
    
    showLoading("Running 3D biometric calculations...");
    try {
        const res = await fetch(`${API_BASE}/api/estimate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                tree_id: parseInt(treeId),
                dbh_method: dbhMethod,
                dbh_params: dbhParams,
                volume_method: volumeMethod,
                volume_params: volumeParams,
                wood_density_kg_m3: woodDensity
            })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Calculation error.");
        
        const errBanner = document.getElementById("single-tree-error-banner");
        if (errBanner) errBanner.style.display = "none";
        
        // Display metrics
        document.getElementById("result-dbh").innerText = data.dbh_cm ? data.dbh_cm.toFixed(1) : "N/A";
        document.getElementById("result-height").innerText = data.height_m.toFixed(1);
        document.getElementById("result-volume").innerText = data.volume_m3 ? data.volume_m3.toFixed(3) : "N/A";
        document.getElementById("result-mass").innerText = data.mass_kg ? data.mass_kg.toFixed(0) : "N/A";
        
        printLog(`--- Estimation Completed (Tree #${treeId}) ---`, "highlight");
        printLog(`DBH: ${data.dbh_cm ? data.dbh_cm.toFixed(2) + "cm" : "N/A"} | Height: ${data.height_m.toFixed(2)}m`, "log");
        printLog(`Volume: ${data.volume_m3 ? data.volume_m3.toFixed(4) + " m³" : "N/A"}`, "success");
        
        if (data.mesh) {
            renderFittedMesh(data.mesh.vertices, data.mesh.faces, data.mesh.colors);
        } else {
            clearMesh();
            printLog("Warning: No 3D mesh generated.", "comment");
        }
    } catch (err) {
        // Reset metrics to show failure values gracefully
        document.getElementById("result-dbh").innerText = "N/A";
        if (currentTreeData && currentTreeData.bounds) {
            document.getElementById("result-height").innerText = currentTreeData.bounds.height.toFixed(1);
        } else {
            document.getElementById("result-height").innerText = "N/A";
        }
        document.getElementById("result-volume").innerText = "N/A";
        document.getElementById("result-mass").innerText = "N/A";
        
        clearMesh();
        
        printLog(`Estimation Error: ${err.message}`, "error");
        
        // Provide helpful diagnostic comments/hints
        let hintText = "";
        if (err.message.includes("dbh_cm e height_m")) {
            hintText = "DBH calculation returned N/A (failed). Cylinder volume requires a valid DBH. Adjust RANSAC parameters or slice thickness to obtain a DBH.";
            printLog("Hint: " + hintText, "comment");
        } else if (err.message.includes("slices")) {
            hintText = "Axis profile / taper / frustum slice processing found insufficient points. Try increasing slice thickness or lowering minimum point requirements.";
            printLog("Hint: " + hintText, "comment");
        } else {
            hintText = "Try adjusting method parameters or choosing an alternative reconstruction algorithm.";
        }
        
        const errBanner = document.getElementById("single-tree-error-banner");
        if (errBanner) {
            errBanner.style.display = "block";
            errBanner.innerHTML = `<i class="bx bx-error-circle"></i> <strong>Estimation Failed:</strong> ${err.message}<br><small>${hintText}</small>`;
        }
    } finally {
        hideLoading();
    }
}

async function handleRunBatch() {
    if (!currentLoadedFile) return;
    
    const dbhMethod = document.getElementById("dbh-method").value;
    const volumeMethod = document.getElementById("volume-method").value;
    const woodDensity = parseFloat(document.getElementById("wood-density").value) || 900.0;
    
    const dbhParams = getParameters("dbh", dbhMethods[dbhMethod] || []);
    const volumeParams = getParameters("vol", volumeMethods[volumeMethod] || []);
    
    showLoading("Running batch calculations on all plot trees...");
    try {
        const res = await fetch(`${API_BASE}/api/estimate-all`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dbh_method: dbhMethod,
                dbh_params: dbhParams,
                volume_method: volumeMethod,
                volume_params: volumeParams,
                wood_density_kg_m3: woodDensity
            })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Batch calculation failed.");
        
        batchResults = data.results;
        
        // Show sections
        document.getElementById("macro-summary-section").style.display = "block";
        document.getElementById("macro-table-section").style.display = "block";
        
        // Populate Summary
        const s = data.summary;
        document.getElementById("macro-total-volume").innerText = s.total_volume_m3.toFixed(2);
        document.getElementById("macro-mean-dbh").innerText = s.mean_dbh_cm.toFixed(1);
        document.getElementById("macro-mean-height").innerText = s.mean_height_m.toFixed(1);
        
        const biomassTons = (s.total_mass_kg / 1000).toFixed(1);
        document.getElementById("macro-success-count").innerText = `${s.successful_count} / ${biomassTons} t`;
        
        // Render macro inventory table
        renderTable(data.results);
        
        // Render all fitted tree meshes simultaneously!
        renderPlotMeshes(data.results);
        
        printLog(`--- Batch Execution Completed ---`, "success");
        printLog(`Processed ${s.tree_count} trees in plot. Success rate: ${(s.successful_count / s.tree_count * 100).toFixed(0)}%`, "log");
        printLog(`Total volume: ${s.total_volume_m3.toFixed(3)} m³ | Total biomass: ${biomassTons} tons`, "highlight");
    } catch (err) {
        printLog(`Batch Error: ${err.message}`, "error");
        alert(`Batch calculation failed: ${err.message}`);
    } finally {
        hideLoading();
    }
}

function renderTable(results) {
    const tbody = document.getElementById("inventory-table-body");
    tbody.innerHTML = "";
    
    results.forEach(r => {
        const row = document.createElement("tr");
        if (r.status === "failed") {
            row.className = "row-failed";
            row.title = `Error: ${r.error}`;
        }
        
        const dbhText = r.dbh_cm ? r.dbh_cm.toFixed(1) : "-";
        const hText = r.height_m ? r.height_m.toFixed(1) : "-";
        const volText = r.volume_m3 ? r.volume_m3.toFixed(3) : "-";
        
        row.innerHTML = `
            <td>#${r.tree_id}</td>
            <td>${dbhText}</td>
            <td>${hText}</td>
            <td>${volText}</td>
        `;
        
        row.addEventListener("click", () => {
            document.querySelectorAll(".inventory-table tbody tr").forEach(tr => tr.classList.remove("active-row"));
            row.classList.add("active-row");
            
            // Switch value in controls selector and focus Single tab
            document.getElementById("tree-select").value = r.tree_id;
            document.getElementById("tab-btn-single").click();
            
            handleTreeChange();
        });
        
        tbody.appendChild(row);
    });
}

function handleTableSearch(e) {
    const query = e.target.value.toLowerCase().trim();
    const rows = document.querySelectorAll("#inventory-table-body tr");
    
    rows.forEach(row => {
        const idCell = row.cells[0].textContent.toLowerCase();
        if (idCell.includes(query) || query === "") {
            row.style.display = "";
        } else {
            row.style.display = "none";
        }
    });
}

// =====================================================================
// THREE.JS VIEWPORT RENDERING
// =====================================================================
function initThreeJS() {
    try {
        const container = document.getElementById("canvas-container");
        
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0d16);
        
        camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.05, 1000);
        camera.position.set(0, 30, 40);
        
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        
        container.innerHTML = "";
        container.appendChild(renderer.domElement);
        
        // Recreate overlay container dynamically to prevent container.innerHTML from wiping it out
        const overlay = document.createElement("div");
        overlay.id = "labels-overlay-container";
        overlay.style.position = "absolute";
        overlay.style.top = "0";
        overlay.style.left = "0";
        overlay.style.width = "100%";
        overlay.style.height = "100%";
        overlay.style.pointerEvents = "none";
        overlay.style.overflow = "hidden";
        overlay.style.zIndex = "10";
        container.appendChild(overlay);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
        dirLight1.position.set(20, 50, 30);
        scene.add(dirLight1);
        
        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dirLight2.position.set(-20, -10, -30);
        scene.add(dirLight2);
        
        // Orbit Controls (with error fallback check)
        if (typeof THREE.OrbitControls === "function") {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = true;
        } else {
            console.warn("THREE.OrbitControls not defined on global THREE. Disabling damping.");
        }
        
        window.addEventListener("resize", () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
        
        document.getElementById("canvas-toolbar").style.display = "flex";
        isThreeJSActive = true;
        
        function animate() {
            requestAnimationFrame(animate);
            if (controls) controls.update();
            renderer.render(scene, camera);
            if (typeof updateHTMLLabels === "function") {
                updateHTMLLabels();
            }
        }
        animate();
    } catch (err) {
        console.error("Failed to initialize WebGL view:", err);
        printLog(`WebGL Warning: 3D viewport failed to load (${err.message}). Diagnostics are still fully functional.`, "error");
        isThreeJSActive = false;
    }
}

function handleShowPlot() {
    if (!currentMetadata) return;
    
    // Reset tree selector in sidebar
    document.getElementById("tree-select").value = "";
    document.getElementById("btn-calculate").disabled = true;
    
    // Reset current tree state
    currentTreeData = null;
    clearMesh();
    clearMetricsDisplay();
    
    // Re-render plot point cloud & plot meshes if they exist
    renderPlotPointCloud(currentMetadata.plot_points, currentMetadata.plot_colors, currentMetadata.plot_is_trunk);
    if (batchResults.length > 0) {
        renderPlotMeshes(batchResults);
    }
    
    printLog("Switched back to full plot macro view.", "comment");
}

function handleColorModeChange() {
    if (currentTreeData) {
        renderPointCloud(currentTreeData.points, currentTreeData.colors, currentTreeData.is_trunk);
    } else if (currentMetadata) {
        renderPlotPointCloud(currentMetadata.plot_points, currentMetadata.plot_colors, currentMetadata.plot_is_trunk);
    }
}

function renderPlotPointCloud(points, colors, isTrunk) {
    if (!isThreeJSActive || !scene) return;
    if (activePointCloudObj) scene.remove(activePointCloudObj);
    
    const colorMode = document.getElementById("color-mode-select").value;
    const plotTreeIds = currentMetadata ? currentMetadata.plot_tree_ids : null;
    
    let filteredPoints = [];
    let filteredColors = [];
    
    for (let i = 0; i < points.length; i++) {
        const isTrunkPt = isTrunk && isTrunk[i] === 1;
        if (colorMode === "trunk" && isTrunk && !isTrunkPt) {
            continue; // Skip canopy points in trunk-only mode
        }
        
        filteredPoints.push(points[i][0], points[i][1], points[i][2]);
        
        let r, g, b;
        if (colorMode === "instance" && plotTreeIds) {
            const ptTreeId = plotTreeIds[i];
            const tColor = getTreeColor(ptTreeId);
            r = tColor[0] / 255;
            g = tColor[1] / 255;
            b = tColor[2] / 255;
        } else if (colorMode === "class" && isTrunk) {
            // Semantic segregation colors: orange/brown for trunk, green for leaves
            if (isTrunkPt) {
                r = 210 / 255;
                g = 105 / 255;
                b = 30 / 255;
            } else {
                r = 34 / 255;
                g = 139 / 255;
                b = 34 / 255;
            }
        } else {
            if (colors && colors[i]) {
                r = colors[i][0] / 255;
                g = colors[i][1] / 255;
                b = colors[i][2] / 255;
            } else {
                // Plot heights gradient (green to teal)
                const z = points[i][2];
                const zNorm = Math.min(1.0, Math.max(0.0, z / 30.0));
                r = 0.1;
                g = 0.3 + 0.6 * zNorm;
                b = 0.4 + 0.5 * (1.0 - zNorm);
            }
        }
        filteredColors.push(r, g, b);
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(filteredPoints, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(filteredColors, 3));
    
    const pSize = parseFloat(document.getElementById("point-size-slider").value) / 100;
    const material = new THREE.PointsMaterial({
        size: pSize,
        vertexColors: true,
        transparent: true,
        opacity: 0.75
    });
    
    activePointCloudObj = new THREE.Points(geometry, material);
    scene.add(activePointCloudObj);
    
    // Reset camera viewing the entire plot
    if (controls) {
        controls.target.set(0, 0, 10);
        camera.position.set(0, -50, 40);
        controls.update();
    }
}

function renderPointCloud(points, colors, isTrunk) {
    if (!isThreeJSActive || !scene) return;
    if (activePointCloudObj) scene.remove(activePointCloudObj);
    
    const colorMode = document.getElementById("color-mode-select").value;
    
    let filteredPoints = [];
    let filteredColors = [];
    
    for (let i = 0; i < points.length; i++) {
        const isTrunkPt = isTrunk[i] === 1;
        if (colorMode === "trunk" && !isTrunkPt) {
            continue; // Skip canopy points in trunk-only mode
        }
        
        filteredPoints.push(points[i][0], points[i][1], points[i][2]);
        
        let r, g, b;
        if (colorMode === "instance" && currentTreeData) {
            const tColor = getTreeColor(currentTreeData.tree_id);
            r = tColor[0] / 255;
            g = tColor[1] / 255;
            b = tColor[2] / 255;
        } else if (colorMode === "class") {
            // Semantic segregation colors: orange/brown for trunk, green for leaves
            if (isTrunkPt) {
                r = 210 / 255;
                g = 105 / 255;
                b = 30 / 255;
            } else {
                r = 34 / 255;
                g = 139 / 255;
                b = 34 / 255;
            }
        } else {
            if (colors && colors[i]) {
                r = colors[i][0] / 255;
                g = colors[i][1] / 255;
                b = colors[i][2] / 255;
            } else {
                // Height-based gradient (Teal to green)
                const z = points[i][2];
                const zNorm = Math.min(1.0, Math.max(0.0, z / 25.0));
                r = 0.1;
                g = 0.4 + 0.5 * zNorm;
                b = 0.6 - 0.2 * zNorm;
            }
        }
        filteredColors.push(r, g, b);
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(filteredPoints, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(filteredColors, 3));
    
    const pSize = parseFloat(document.getElementById("point-size-slider").value) / 100;
    const material = new THREE.PointsMaterial({
        size: pSize,
        vertexColors: true,
        transparent: true,
        opacity: 0.85
    });
    
    activePointCloudObj = new THREE.Points(geometry, material);
    scene.add(activePointCloudObj);
    
    resetCamera();
}

function renderFittedMesh(vertices, faces, meshColors) {
    if (!isThreeJSActive || !scene) return;
    clearMesh();
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices.flat(), 3));
    
    if (meshColors) {
        const floatColors = meshColors.map(c => c.map(val => val / 255));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(floatColors.flat(), 3));
    }
    
    geometry.setIndex(faces.flat());
    geometry.computeVertexNormals();
    
    const opacityVal = parseFloat(document.getElementById("mesh-opacity-slider").value) / 100;
    const isVisible = document.getElementById("toggle-mesh").classList.contains("active");
    
    const material = new THREE.MeshStandardMaterial({
        vertexColors: !meshColors ? false : true,
        color: !meshColors ? 0x06b6d4 : undefined,
        transparent: true,
        opacity: opacityVal,
        roughness: 0.5,
        metalness: 0.2,
        side: THREE.DoubleSide
    });
    
    activeMeshObj = new THREE.Mesh(geometry, material);
    activeMeshObj.visible = isVisible;
    scene.add(activeMeshObj);
}

function renderPlotMeshes(results) {
    if (!isThreeJSActive || !scene) return;
    clearMesh();
    
    let combinedVertices = [];
    let combinedFaces = [];
    let combinedColors = [];
    let vertexOffset = 0;
    
    results.forEach((r, idx) => {
        if (r.status === "success" && r.mesh) {
            const v = r.mesh.vertices;
            const f = r.mesh.faces;
            const c = r.mesh.colors;
            
            combinedVertices.push(...v.flat());
            
            const adjustedFaces = f.map(face => face.map(idx => idx + vertexOffset));
            combinedFaces.push(...adjustedFaces.flat());
            
            if (c) {
                const floatColors = c.map(val => val.map(val2 => val2 / 255));
                combinedColors.push(...floatColors.flat());
            } else {
                // Generate a distinctive pastel color for each tree cylinder
                const rColor = 0.2 + 0.6 * ((idx * 17) % 10 / 10);
                const gColor = 0.2 + 0.6 * ((idx * 23) % 10 / 10);
                const bColor = 0.2 + 0.6 * ((idx * 29) % 10 / 10);
                for (let i = 0; i < v.length; i++) {
                    combinedColors.push(rColor, gColor, bColor);
                }
            }
            vertexOffset += v.length;
        }
    });
    
    if (combinedVertices.length === 0) return;
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(combinedVertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(combinedColors, 3));
    geometry.setIndex(combinedFaces);
    geometry.computeVertexNormals();
    
    const opacityVal = parseFloat(document.getElementById("mesh-opacity-slider").value) / 100;
    const isVisible = document.getElementById("toggle-mesh").classList.contains("active");
    
    const material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        transparent: true,
        opacity: opacityVal,
        roughness: 0.5,
        metalness: 0.2,
        side: THREE.DoubleSide
    });
    
    activeMeshObj = new THREE.Mesh(geometry, material);
    activeMeshObj.visible = isVisible;
    scene.add(activeMeshObj);
}

function clearMesh() {
    if (!isThreeJSActive || !scene) return;
    if (activeMeshObj) {
        scene.remove(activeMeshObj);
        activeMeshObj = null;
    }
}

function resetCamera() {
    if (!isThreeJSActive || !controls) return;
    
    if (currentTreeData && currentTreeData.bounds) {
        const bounds = currentTreeData.bounds;
        const height = bounds.height;
        controls.target.set(0, 0, height / 2);
        camera.position.set(0, -height * 1.5, height);
        controls.update();
    } else if (currentMetadata) {
        controls.target.set(0, 0, 10);
        camera.position.set(0, -50, 40);
        controls.update();
    }
}

// =====================================================================
// HELPER UTILITIES
// =====================================================================
function printLog(msg, type = "log") {
    const term = document.getElementById("diagnostic-body");
    const line = document.createElement("div");
    
    if (type === "error") {
        line.innerHTML = `<span class="terminal-error">[ERROR] ${msg}</span>`;
    } else if (type === "success") {
        line.innerHTML = `<span class="terminal-success">[SUCCESS] ${msg}</span>`;
    } else if (type === "highlight") {
        line.innerHTML = `<span class="terminal-highlight">[INFO] ${msg}</span>`;
    } else if (type === "comment") {
        line.innerHTML = `<span class="terminal-comment">${msg}</span>`;
    } else {
        line.innerHTML = `<span class="terminal-log">${msg}</span>`;
    }
    
    term.appendChild(line);
    term.scrollTop = term.scrollHeight;
}

function showLoading(msg) {
    const overlay = document.getElementById("loading-overlay");
    document.getElementById("loading-message").innerText = msg;
    overlay.style.display = "flex";
}

function hideLoading() {
    document.getElementById("loading-overlay").style.display = "none";
}

function clearMetricsDisplay() {
    document.getElementById("result-dbh").innerText = "-";
    document.getElementById("result-height").innerText = "-";
    document.getElementById("result-volume").innerText = "-";
    document.getElementById("result-mass").innerText = "-";
}

function PathBasename(path) {
    return path.split(/[\\/]/).pop();
}

function getTreeColor(treeId) {
    if (treeId === undefined || treeId === null || treeId < 0) {
        return [128, 128, 128]; // Default gray for unassigned points
    }
    // Knuth's multiplicative hash to disperse colors vibrantly
    let hash = (treeId * 2654435761) & 0xFFFFFFFF;
    let r = (hash & 0xFF0000) >> 16;
    let g = (hash & 0x00FF00) >> 8;
    let b = (hash & 0x0000FF);
    
    // Scale and normalize colors to keep them vibrant and clear against the dark background (0x0a0d16)
    let max = Math.max(r, g, b, 1);
    r = Math.floor(60 + (r / max) * 195);
    g = Math.floor(60 + (g / max) * 195);
    b = Math.floor(60 + (b / max) * 195);
    
    return [r, g, b];
}

function updateHTMLLabels() {
    if (!isThreeJSActive || !scene || !camera) return;
    
    const labelContainer = document.getElementById("labels-overlay-container");
    if (!labelContainer) return;
    
    const showLabels = document.getElementById("toggle-labels") && document.getElementById("toggle-labels").classList.contains("active");
    if (!showLabels || !currentMetadata || !currentMetadata.tree_centers) {
        labelContainer.innerHTML = "";
        return;
    }
    
    // If a single tree is active (Single inspect view), do not show macro labels to avoid confusion
    if (currentTreeData) {
        labelContainer.innerHTML = "";
        return;
    }
    
    // Project and position each tree label based on its max Z (top center) relative coordinates
    const width = labelContainer.clientWidth;
    const height = labelContainer.clientHeight;
    
    const statuses = {};
    batchResults.forEach(r => {
        statuses[r.tree_id] = r.status;
    });
    
    let html = "";
    
    for (const [tid, center] of Object.entries(currentMetadata.tree_centers)) {
        const vec = new THREE.Vector3(center[0], center[1], center[2]);
        vec.project(camera);
        
        if (vec.z > 1) continue; // Behind camera
        
        const x = (vec.x * 0.5 + 0.5) * width;
        const y = (-(vec.y * 0.5) + 0.5) * height;
        
        const status = statuses[tid];
        let labelClass = "tree-label-neutral";
        let statusText = "";
        
        if (status === "success") {
            labelClass = "tree-label-success";
            statusText = " &bull; Success";
        } else if (status === "failed") {
            labelClass = "tree-label-failed";
            statusText = " &bull; Failed";
        }
        
        html += `<div class="tree-3d-label ${labelClass}" style="position: absolute; left: ${x}px; top: ${y}px; transform: translate(-50%, -100%);">
            #${tid}${statusText}
        </div>`;
    }
    
    labelContainer.innerHTML = html;
}
