# BoreSight-Q 💧🛰️⚛️

**Parcel-Scale Hydrogeological Borewell Siting Engine powered by Hybrid Quantum Variational Regression (QPU) & Multi-Sensor Satellite Data**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum_VQR-blueviolet.svg?style=flat)](https://pennylane.ai)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?style=flat&logo=React&logoColor=black)](https://react.dev)
[![MapLibre GL](https://img.shields.io/badge/MapLibre_GL-4.7-blue.svg?style=flat)](https://maplibre.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC.svg?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)

---

## 📌 Overview

**BoreSight-Q** is an enterprise-grade hydrogeological and geophysical exploration platform designed to identify optimal borehole drilling locations (P1 primary, P2 secondary, and P3 contingency) for agricultural parcels and hard-rock terrains. 

By unifying **Copernicus Sentinel-1 SAR**, **Sentinel-2 multispectral imagery**, **Copernicus 30m Global DEM**, **multi-temporal InSAR ground deformation velocities**, and **ISRIC SoilGrids hydraulic conductivity** with a **12-Qubit Hybrid Variational Quantum Regressor (PennyLane + PyTorch)** and an **Analytical Hierarchy Process (AHP)** physics prior, BoreSight-Q delivers high-precision groundwater potential mapping at parcel scale.

---

## ⚡ Core Capabilities

1. **Multi-Sensor Satellite Harmonization**:
   - **Sentinel-1 C-Band SAR**: VV/VH backscatter extraction and dielectric soil moisture anomaly detection.
   - **Sentinel-2 Multispectral**: High-resolution optical bands for NDVI (vegetation vigor), NDMI (canopy moisture), and NDWI/MNDWI (surface water rejection).
   - **Copernicus DEM (30m)**: Topographic Wetness Index (TWI), slope gradients, profile/plan curvatures, and Strahler stream network extraction.
   - **ISRIC SoilGrids & Multi-temporal InSAR**: Saturated hydraulic conductivity ($K_{sat}$) and millimeter-scale aquifer compaction/rebound velocity.

2. **12-Qubit Hybrid Variational Quantum Regressor (VQR)**:
   - 12-wire quantum angle embedding ($R_y$) with physics-informed Ising $ZZ$ entangling gates between hydrogeologically coupled observables (e.g., TWI $\leftrightarrow$ SAR moisture, Drainage distance $\leftrightarrow$ Lineament intersection distance).
   - Ensemble fusion combining 35% Variational Quantum Regressor output with 65% calibrated AHP physics prior vector.

3. **Intelligent Borehole Siting & Hydraulic Sizing**:
   - Automated non-maximum suppression (NMS) siting of **P1 (Optimal)**, **P2 (Secondary)**, and **P3 (Contingency)** targets strictly inside the parcel boundary.
   - Exact hydraulic delivery pipe sizing ($1.0''$, $1.25''$, $1.5''$, $2.0''$, $2.5''$) calibrated to standard borehole flow velocity ($1.2 - 2.1\text{ m/s}$) and submersible pump horsepower (HP) rating based on estimated sustainable discharge (LPH).

4. **WGS84 Geodesic Survey Engine**:
   - Real-time ellipsoidal geodesic parcel boundary measurements with $<0.001\%$ parcel-scale distortion.
   - Comprehensive unit readouts in **Acres & Cents**, **Hectares**, **Square Meters**, **Square Yards**, and **Perimeter (m / ft)**.

5. **Export & Field Deployment**:
   - One-click Google Earth `.kml` export containing geodesic parcel boundary polygon, structural lineaments, drainage networks, and rich HTML specification balloons for P1, P2, and P3 drilling pins.

---

## 🏗️ Architecture

```
HydroMark2/
├── backend/
│   ├── main.py                     # FastAPI application, KML export & raster generators
│   ├── schemas.py                  # Pydantic v2 data models & validation
│   ├── requirements.txt            # Python dependencies (PennyLane, PyTorch, GDAL, Shapely, etc.)
│   ├── models/
│   │   └── quantum_vqr.py          # 12-Qubit PennyLane VQR, AHP Prior & Pipe Sizing
│   └── services/
│       ├── hydro_geology.py        # D8 drainage flow, lineament extraction & curvature
│       ├── insar_velocity.py       # InSAR deformation & SoilGrids hydraulic conductivity
│       └── satellite_fetcher.py    # Planetary Computer Sentinel-1/2 & DEM co-registration
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Application state container & layout
│   │   ├── components/
│   │   │   ├── GeologistMap.jsx    # MapLibre GL satellite map with interactive parcel drawing
│   │   │   ├── BorewellReport.jsx  # Hydrogeological consultation card & target specs
│   │   │   └── Header.jsx          # Top navigation & toolbar
│   │   ├── utils/
│   │   │   └── geodesicArea.js     # WGS84 geodesic polygon calculation engine
│   │   └── data/
│   │       └── demoParcels.js      # Reference agricultural test parcels
│   ├── vite.config.js              # Vite server & proxy configuration
│   └── tailwind.config.js          # Styling & color systems
└── tests/
    ├── test_api.py                 # API endpoint integration test suite
    └── test_parcel_verification.py # Hydrogeological pipeline test suite
```

---

## 🚀 Quickstart Guide

### Prerequisites
- **Python 3.10+** (with virtual environment)
- **Node.js 18+** & **npm**

---

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/Raghavv7989/BoreSight-Q.git
cd BoreSight-Q

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Start FastAPI server on port 8000
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be accessible at:
- **API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start Vite dev server on port 5173
npm run dev -- --host 0.0.0.0 --port 5173
```

Frontend application will open at: [http://localhost:5173](http://localhost:5173).

---

## 🛰️ API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check for satellite fetcher, InSAR, hydrogeology & QPU services |
| `POST` | `/api/analyze-parcel` | End-to-end hydrogeological analysis, VQR inference, and target siting |
| `POST` | `/api/export-kml` | Generates 3D Google Earth KML package with styled placemarks & balloons |

---

## 📜 License

This project is licensed under the MIT License.
