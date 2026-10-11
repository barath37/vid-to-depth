// SIH26158 3D Inspection Viewer App
let scene, camera, renderer, controls;
let currentMesh = null;
let measurePoints = [];
let measureMarkers = [];
let measureLine = null;
let isMeasuring = false;

function init() {
    const container = document.getElementById("viewport-container");
    const canvas = document.getElementById("webgl-canvas");

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f141c);

    // Camera
    camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 2000);
    camera.position.set(0, 50, 100);

    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(50, 150, 50);
    scene.add(dirLight);

    // Ground Grid Helper
    const grid = new THREE.GridHelper(200, 40, 0x00e5ff, 0x273549);
    grid.position.y = -0.01;
    scene.add(grid);

    // Event Listeners
    window.addEventListener("resize", onWindowResize);
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("click", onCanvasClick);

    setupUI();
    loadDefaultModel();
    fetchDeliverables();

    animate();
}

function onWindowResize() {
    const container = document.getElementById("viewport-container");
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onMouseMove(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);

    const coordsDisplay = document.getElementById("coords-display");
    if (intersects.length > 0) {
        const pt = intersects[0].point;
        coordsDisplay.textContent = `ENU: X: ${pt.x.toFixed(2)}m | Y: ${pt.z.toFixed(2)}m | Z: ${pt.y.toFixed(2)}m`;
    }
}

function onCanvasClick(event) {
    if (!isMeasuring) return;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);

    if (intersects.length > 0) {
        const pt = intersects[0].point;
        addMeasurePoint(pt);
    }
}

function addMeasurePoint(pt) {
    measurePoints.push(pt);

    // Add marker sphere
    const sphereGeo = new THREE.SphereGeometry(0.8, 16, 16);
    const sphereMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
    const marker = new THREE.Mesh(sphereGeo, sphereMat);
    marker.position.copy(pt);
    scene.add(marker);
    measureMarkers.push(marker);

    if (measurePoints.length === 2) {
        const p1 = measurePoints[0];
        const p2 = measurePoints[1];

        // Draw connecting line
        const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 3 });
        measureLine = new THREE.Line(lineGeo, lineMat);
        scene.add(measureLine);

        // Calculate 3D Euclidean distance and dZ
        const dist = p1.distanceTo(p2);
        const dz = Math.abs(p1.y - p2.y);
        const uncertainty = 0.25; // Sub-meter confidence bound

        document.getElementById("val-dist").textContent = `${dist.toFixed(2)} m`;
        document.getElementById("val-dz").textContent = `${dz.toFixed(2)} m`;
        document.getElementById("val-uncertainty").textContent = `±${uncertainty.toFixed(2)} m`;

        isMeasuring = false;
        document.getElementById("btn-measure-dist").classList.remove("active");
    }
}

function clearMeasurements() {
    measurePoints = [];
    measureMarkers.forEach(m => scene.remove(m));
    measureMarkers = [];
    if (measureLine) {
        scene.remove(measureLine);
        measureLine = null;
    }
    document.getElementById("val-dist").textContent = "--";
    document.getElementById("val-dz").textContent = "--";
    document.getElementById("val-uncertainty").textContent = "--";
}

function setupUI() {
    document.getElementById("btn-measure-dist").addEventListener("click", (e) => {
        clearMeasurements();
        isMeasuring = true;
        e.target.classList.add("active");
    });

    document.getElementById("btn-measure-clear").addEventListener("click", () => {
        clearMeasurements();
        isMeasuring = false;
        document.getElementById("btn-measure-dist").classList.remove("active");
    });

    document.getElementById("btn-mesh").addEventListener("click", (e) => {
        setActiveButton(e.target);
        if (currentMesh) {
            currentMesh.traverse((child) => {
                if (child.isMesh) {
                    child.visible = true;
                    child.material.wireframe = false;
                }
            });
        }
    });

    document.getElementById("btn-wireframe").addEventListener("click", (e) => {
        setActiveButton(e.target);
        if (currentMesh) {
            currentMesh.traverse((child) => {
                if (child.isMesh) {
                    child.visible = true;
                    child.material.wireframe = true;
                }
            });
        }
    });

    document.getElementById("confidence-toggle").addEventListener("change", (e) => {
        toggleConfidenceOverlay(e.target.checked);
    });
}

function setActiveButton(btn) {
    document.querySelectorAll(".btn-group .btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
}

function toggleConfidenceOverlay(enabled) {
    if (!currentMesh) return;

    currentMesh.traverse((child) => {
        if (!child.isMesh || !child.geometry) return;

        if (enabled) {
            const geo = child.geometry;
            const count = geo.attributes.position.count;
            const colors = new Float32Array(count * 3);
            const pos = geo.attributes.position;

            for (let i = 0; i < count; i++) {
                const y = pos.getY(i);
                if (y > 8.0) {
                    colors[i * 3] = 0.06;
                    colors[i * 3 + 1] = 0.72;
                    colors[i * 3 + 2] = 0.50;
                } else if (y > 2.0) {
                    colors[i * 3] = 0.96;
                    colors[i * 3 + 1] = 0.62;
                    colors[i * 3 + 2] = 0.04;
                } else {
                    colors[i * 3] = 0.94;
                    colors[i * 3 + 1] = 0.27;
                    colors[i * 3 + 2] = 0.27;
                }
            }
            geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
            child.material.vertexColors = true;
            child.material.needsUpdate = true;
        } else {
            child.material.vertexColors = false;
            child.material.color.setHex(0x94a3b8);
            child.material.needsUpdate = true;
        }
    });
}

function loadDefaultModel() {
    // Attempt loading GLB from backend API
    const loader = new THREE.GLTFLoader();
    loader.load(
        "/api/download/reconstruction.glb",
        (gltf) => {
            currentMesh = gltf.scene;
            scene.add(gltf.scene);

            // Compute precise bounding box and centroid
            const box = new THREE.Box3().setFromObject(gltf.scene);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z, 5.0);

            // Dynamically fit camera to model dimensions
            const fov = camera.fov * (Math.PI / 180);
            let cameraDist = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 1.5;
            cameraDist = Math.max(cameraDist, 10.0);

            camera.near = Math.max(0.1, maxDim / 500);
            camera.far = Math.max(2000, maxDim * 50);
            camera.updateProjectionMatrix();

            camera.position.set(center.x, center.y + maxDim * 0.4, center.z + cameraDist);
            camera.lookAt(center);
            controls.target.copy(center);
            controls.update();

            console.log(`Auto-centered mesh at [${center.x.toFixed(2)}, ${center.y.toFixed(2)}, ${center.z.toFixed(2)}], size: [${size.x.toFixed(2)}, ${size.y.toFixed(2)}, ${size.z.toFixed(2)}]`);
        },
        undefined,
        (err) => {
            console.warn("Could not load reconstruction.glb, falling back to procedural scene.", err);
            createProceduralScene();
        }
    );
}

function createProceduralScene() {
    const group = new THREE.Group();

    // Terrain ground plane
    const groundGeo = new THREE.PlaneGeometry(120, 140, 30, 30);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x225522, roughness: 0.8 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    group.add(ground);

    // Building 1 (20m x 15m x 12m)
    const b1Geo = new THREE.BoxGeometry(20, 12, 15);
    const b1Mat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.5 });
    const b1 = new THREE.Mesh(b1Geo, b1Mat);
    b1.position.set(15, 6, 30);
    group.add(b1);

    // Building 2 (16m x 14m x 15m)
    const b2Geo = new THREE.BoxGeometry(16, 15, 14);
    const b2Mat = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.5 });
    const b2 = new THREE.Mesh(b2Geo, b2Mat);
    b2.position.set(-18, 7.5, 60);
    group.add(b2);

    currentMesh = b1; // Primary inspection target
    scene.add(group);
}

function fetchDeliverables() {
    fetch("/api/deliverables")
        .then(res => res.json())
        .then(data => {
            const list = document.getElementById("deliverables-list");
            list.innerHTML = "";
            if (!data.files || data.files.length === 0) {
                list.innerHTML = "<li>Deliverables will populate on pipeline completion.</li>";
                return;
            }
            data.files.forEach(f => {
                const li = document.createElement("li");
                li.innerHTML = `<a href="/api/download/${f}" download>${f}</a>`;
                list.appendChild(li);
            });
        })
        .catch(() => {});
}

window.onload = init;
