import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import maplibregl from 'maplibre-gl';
import {
  Layers,
  Eye,
  EyeOff,
  Navigation,
  Compass,
  Search,
  Crosshair,
  MapPin,
  Trash2,
  CheckCircle,
  Play,
  Activity,
  Sliders,
  ChevronDown,
  RotateCcw,
  AlertTriangle
} from 'lucide-react';
import { DEMO_PARCELS } from '../data/demoParcels';
import { calculateGeodesicLandMetrics, computeSegmentDistanceM } from '../utils/geodesicArea';

const ESRI_SATELLITE_STYLE = {
  version: 8,
  glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
  sources: {
    'esri-satellite': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      attribution: 'Tiles &copy; Esri, Maxar, Earthstar Geographics',
      maxzoom: 19,
    },
  },
  layers: [
    {
      id: 'esri-satellite-layer',
      type: 'raster',
      source: 'esri-satellite',
      minzoom: 0,
      maxzoom: 19,
    },
  ],
};

export default function GeologistMap({
  analysisData,
  isLoading,
  onAnalyze,
  selectedPoint,
  onSelectPoint,
  parcelPolygon,
  onParcelChange,
  onClearSurvey,
}) {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef([]);
  const popupRef = useRef(null);
  const centroidMarkerRef = useRef(null);

  // Map readiness flag
  const [isMapLoaded, setIsMapLoaded] = useState(false);
  const pendingBoundaryRef = useRef(null);

  // Drawing state & refs to eliminate stale closures
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawingPoints, setDrawingPoints] = useState([]);
  const isDrawingRef = useRef(false);
  const drawingPointsRef = useRef([]);
  const boundaryLockedRef = useRef(false);

  // Live WGS84 Geodesic Land Area & Perimeter Metrics
  const activeCoords = (isDrawing && drawingPoints.length >= 3)
    ? drawingPoints
    : (parcelPolygon?.coordinates?.[0] || []);

  const currentMetrics = useMemo(() => {
    if (!activeCoords || activeCoords.length < 3) return null;
    return calculateGeodesicLandMetrics(activeCoords);
  }, [activeCoords]);

  // Keep refs synchronized with state
  useEffect(() => {
    isDrawingRef.current = isDrawing;
  }, [isDrawing]);

  useEffect(() => {
    drawingPointsRef.current = drawingPoints;
  }, [drawingPoints]);

  // Search state
  const [searchLat, setSearchLat] = useState('13.4350');
  const [searchLon, setSearchLon] = useState('77.7275');

  // Layer Visibility & Settings
  const [showDrainage, setShowDrainage] = useState(true);
  const [showLineaments, setShowLineaments] = useState(true);
  const [showIntersections, setShowIntersections] = useState(true);
  const [activeRaster, setActiveRaster] = useState('gwpi'); // 'gwpi' | 'sar' | 'insar' | 'none'
  const [rasterOpacity, setRasterOpacity] = useState(0.80);
  const [showLayerMenu, setShowLayerMenu] = useState(true);

  // Helper: Ensure parcel boundary source and progressive layers exist on the map
  const ensureParcelLayers = useCallback((map) => {
    if (!map || !map.isStyleLoaded()) return;

    try {
      if (!map.getSource('parcel-boundary')) {
        map.addSource('parcel-boundary', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
        });
      }

      // 1. Polygon Fill Layer (3+ points) - 0.15 fill opacity
      if (!map.getLayer('parcel-fill')) {
        map.addLayer({
          id: 'parcel-fill',
          type: 'fill',
          source: 'parcel-boundary',
          filter: ['==', ['geometry-type'], 'Polygon'],
          paint: {
            'fill-color': '#FFD700',
            'fill-opacity': 0.15,
          },
        });
      }

      // 2. Yellow LineString Layer (2+ points) - bright yellow outline, line-width: 3.5
      if (!map.getLayer('parcel-line')) {
        map.addLayer({
          id: 'parcel-line',
          type: 'line',
          source: 'parcel-boundary',
          filter: ['in', ['geometry-type'], 'LineString', 'Polygon'],
          paint: {
            'line-color': '#FFD700',
            'line-width': 3.5,
          },
        });
      }

      // 3. Yellow Vertex Circles (1+ points)
      if (!map.getLayer('parcel-vertices')) {
        map.addLayer({
          id: 'parcel-vertices',
          type: 'circle',
          source: 'parcel-boundary',
          filter: ['==', ['geometry-type'], 'Point'],
          paint: {
            'circle-radius': 6,
            'circle-color': '#FFD700',
            'circle-stroke-width': 2,
            'circle-stroke-color': '#000000',
          },
        });
      }

      // 4. Vertex Index Labels (1, 2, 3...)
      if (!map.getLayer('parcel-vertex-labels')) {
        map.addLayer({
          id: 'parcel-vertex-labels',
          type: 'symbol',
          source: 'parcel-boundary',
          filter: ['==', ['geometry-type'], 'Point'],
          layout: {
            'text-field': ['to-string', ['get', 'index']],
            'text-size': 11,
            'text-offset': [0, -1.2],
            'text-allow-overlap': true,
            'text-ignore-placement': true,
          },
          paint: {
            'text-color': '#FFD700',
            'text-halo-color': '#000000',
            'text-halo-width': 2,
          },
        });
      }

      // 5. Geodesic Segment Distance Source & Layer
      if (!map.getSource('parcel-measurements-src')) {
        map.addSource('parcel-measurements-src', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
        });
      }

      if (!map.getLayer('parcel-segment-labels')) {
        map.addLayer({
          id: 'parcel-segment-labels',
          type: 'symbol',
          source: 'parcel-measurements-src',
          filter: ['==', ['get', 'type'], 'segment'],
          layout: {
            'text-field': ['get', 'label'],
            'text-size': 10,
            'text-offset': [0, 0],
            'text-allow-overlap': true,
            'text-ignore-placement': true,
          },
          paint: {
            'text-color': '#FFFFFF',
            'text-halo-color': '#000000',
            'text-halo-width': 2.5,
          },
        });
      }
    } catch (e) {
      console.warn('ensureParcelLayers warning:', e);
    }
  }, []);

  // Helper: Progressive GeoJSON FeatureCollection builder
  const renderParcelBoundary = useCallback((points, isClosed = false) => {
    const map = mapRef.current;
    if (!map) return;

    if (!map.isStyleLoaded()) {
      pendingBoundaryRef.current = { points, isClosed };
      return;
    }

    try {
      ensureParcelLayers(map);
      const src = map.getSource('parcel-boundary');
      if (!src) return;

      if (!points || points.length === 0) {
        src.setData({ type: 'FeatureCollection', features: [] });
        const measSrc = map.getSource('parcel-measurements-src');
        if (measSrc) measSrc.setData({ type: 'FeatureCollection', features: [] });
        if (centroidMarkerRef.current) {
          centroidMarkerRef.current.remove();
          centroidMarkerRef.current = null;
        }
        return;
      }

      const features = [];

      // 1+ points: Vertex Points
      points.forEach((pt, idx) => {
        features.push({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: pt },
          properties: { index: idx + 1 },
        });
      });

      // 2+ points: LineString
      if (points.length >= 2) {
        const lineCoords = (points.length >= 3 || isClosed)
          ? [...points, points[0]]
          : points;
        features.push({
          type: 'Feature',
          geometry: { type: 'LineString', coordinates: lineCoords },
          properties: {},
        });
      }

      // 3+ points: Closed Polygon
      if (points.length >= 3) {
        const ring = [...points, points[0]];
        features.push({
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [ring] },
          properties: {},
        });
      }

      src.setData({ type: 'FeatureCollection', features });

      // Update Segment Distance Labels at Edge Midpoints
      const measSrc = map.getSource('parcel-measurements-src');
      if (measSrc) {
        if (points.length < 2) {
          measSrc.setData({ type: 'FeatureCollection', features: [] });
        } else {
          const measFeatures = [];
          const isRing = (points.length >= 3 || isClosed);
          const n = points.length;
          const numEdges = isRing ? n : n - 1;
          for (let i = 0; i < numEdges; i++) {
            const pA = points[i];
            const pB = points[(i + 1) % n];
            const segLenM = computeSegmentDistanceM(pA, pB);
            const segLenFt = Math.round(segLenM * 3.280839895);
            const mid = [(pA[0] + pB[0]) / 2, (pA[1] + pB[1]) / 2];
            measFeatures.push({
              type: 'Feature',
              geometry: { type: 'Point', coordinates: mid },
              properties: {
                type: 'segment',
                label: `${segLenM.toFixed(1)} m (${segLenFt} ft)`,
              },
            });
          }
          measSrc.setData({ type: 'FeatureCollection', features: measFeatures });
        }
      }

      // Floating Centroid Area Badge
      if (points.length >= 3) {
        const metrics = calculateGeodesicLandMetrics(points);
        if (metrics && metrics.centroid && metrics.centroid[0] !== 0) {
          if (!centroidMarkerRef.current) {
            const el = document.createElement('div');
            el.className = 'parcel-centroid-badge pointer-events-none select-none';
            centroidMarkerRef.current = new maplibregl.Marker({ element: el })
              .setLngLat(metrics.centroid)
              .addTo(map);
          } else {
            centroidMarkerRef.current.setLngLat(metrics.centroid);
          }
          centroidMarkerRef.current.getElement().innerHTML = `
            <div style="background: rgba(14, 22, 38, 0.94); border: 1.5px solid #FFD700; border-radius: 8px; padding: 4px 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.65); text-align: center; white-space: nowrap; transform: translate(-50%, -50%);">
              <div style="font-family: monospace; font-size: 13px; font-weight: 800; color: #FFD700; line-height: 1.2;">
                ${metrics.area_acres.toFixed(2)} Acres
              </div>
              <div style="font-size: 10px; color: #94A3B8; font-weight: 600; margin-top: 1px;">
                (${metrics.area_hectares.toFixed(2)} ha | ${Math.round(metrics.area_sq_m).toLocaleString()} m²)
              </div>
            </div>
          `;
        }
      } else {
        if (centroidMarkerRef.current) {
          centroidMarkerRef.current.remove();
          centroidMarkerRef.current = null;
        }
      }
    } catch (err) {
      console.warn('renderParcelBoundary warning:', err);
    }
  }, [ensureParcelLayers]);

  // Initialize Map ONLY ONCE on mount
  useEffect(() => {
    if (!mapContainerRef.current) return;

    const initialCenter = [77.7275, 13.4350]; // Default hard-rock Karnataka terrain
    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: ESRI_SATELLITE_STYLE,
      center: initialCenter,
      zoom: 14.5,
      pitch: 20,
      bearing: 0,
      attributionControl: false,
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: true, visualizePitch: true }), 'top-right');
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: 'metric' }), 'bottom-left');

    const setupLayers = () => {
      try {
        ensureParcelLayers(map);

        // Render queued boundary or initial polygon
        if (pendingBoundaryRef.current) {
          const { points, isClosed } = pendingBoundaryRef.current;
          renderParcelBoundary(points, isClosed);
        } else if (parcelPolygon?.coordinates?.[0]) {
          renderParcelBoundary(parcelPolygon.coordinates[0], true);
        }

        // Drainage Network Source & Layers
        if (!map.getSource('drainage-source')) {
          map.addSource('drainage-source', {
            type: 'geojson',
            data: { type: 'FeatureCollection', features: [] },
          });

          map.addLayer({
            id: 'drainage-casing',
            type: 'line',
            source: 'drainage-source',
            layout: { 'line-join': 'round', 'line-cap': 'round' },
            paint: {
              'line-color': '#004d66',
              'line-width': [
                'interpolate', ['linear'], ['get', 'stream_order'],
                1, 3.5,
                2, 4.5,
                3, 6.0,
                4, 7.5
              ],
              'line-opacity': 0.6,
            },
          });

          map.addLayer({
            id: 'drainage-lines',
            type: 'line',
            source: 'drainage-source',
            layout: { 'line-join': 'round', 'line-cap': 'round' },
            paint: {
              'line-color': '#00E5FF',
              'line-width': [
                'interpolate', ['linear'], ['get', 'stream_order'],
                1, 1.5,
                2, 2.5,
                3, 3.8,
                4, 4.8
              ],
              'line-opacity': 0.95,
            },
          });
        }

        // Structural Fractures / Lineaments Source & Layer
        if (!map.getSource('lineaments-source')) {
          map.addSource('lineaments-source', {
            type: 'geojson',
            data: { type: 'FeatureCollection', features: [] },
          });

          map.addLayer({
            id: 'lineaments-lines',
            type: 'line',
            source: 'lineaments-source',
            layout: { 'line-join': 'miter', 'line-cap': 'square' },
            paint: {
              'line-color': '#00FFCC',
              'line-width': 2.5,
              'line-opacity': 0.9,
              'line-dasharray': [4, 1.5],
            },
          });
        }

        // Fracture-Drainage Intersections Source & Layer
        if (!map.getSource('intersections-source')) {
          map.addSource('intersections-source', {
            type: 'geojson',
            data: { type: 'FeatureCollection', features: [] },
          });

          map.addLayer({
            id: 'intersections-ring',
            type: 'circle',
            source: 'intersections-source',
            paint: {
              'circle-radius': 7,
              'circle-color': 'rgba(0, 229, 255, 0.1)',
              'circle-stroke-width': 2,
              'circle-stroke-color': '#00E5FF',
            },
          });

          map.addLayer({
            id: 'intersections-center',
            type: 'circle',
            source: 'intersections-source',
            paint: {
              'circle-radius': 2.5,
              'circle-color': '#FFFFFF',
            },
          });
        }

        // Lineament Hover Tooltip (Azimuth & Length)
        map.on('mouseenter', 'lineaments-lines', (e) => {
          map.getCanvas().style.cursor = 'pointer';
          if (!e.features?.length) return;
          const feat = e.features[0];
          const az = feat.properties?.azimuth_deg;
          const len = feat.properties?.length_m;
          const id = feat.properties?.lineament_id;

          if (popupRef.current) popupRef.current.remove();
          popupRef.current = new maplibregl.Popup({ closeButton: false, offset: 10 })
            .setLngLat(e.lngLat)
            .setHTML(`
              <div class="text-xs font-mono space-y-1">
                <div class="text-[#00FFCC] font-bold">Fracture Strike #${id}</div>
                <div>Azimuth: <span class="font-bold text-white">${Number(az).toFixed(1)}°</span></div>
                <div>Length: <span class="text-slate-300">${Number(len).toFixed(1)} m</span></div>
              </div>
            `)
            .addTo(map);
        });

        map.on('mouseleave', 'lineaments-lines', () => {
          map.getCanvas().style.cursor = isDrawingRef.current ? 'crosshair' : '';
          if (popupRef.current) {
            popupRef.current.remove();
            popupRef.current = null;
          }
        });
      } catch (err) {
        console.warn('setupLayers warning:', err);
      }
    };

    map.on('load', () => {
      setIsMapLoaded(true);
      setupLayers();
    });

    // Map click handler using isDrawingRef to eliminate stale closures
    map.on('click', (e) => {
      if (!isDrawingRef.current || boundaryLockedRef.current) return;

      const coords = [e.lngLat.lng, e.lngLat.lat];
      setDrawingPoints((prev) => {
        const next = [...prev, coords];
        drawingPointsRef.current = next;
        renderParcelBoundary(next, false);
        return next;
      });
    });

    mapRef.current = map;

    return () => {
      if (centroidMarkerRef.current) {
        centroidMarkerRef.current.remove();
        centroidMarkerRef.current = null;
      }
      if (popupRef.current) popupRef.current.remove();
      map.remove();
    };
  }, []); // Run ONLY ONCE on mount!

  // Set the map canvas cursor to 'crosshair' whenever drawing mode is active
  useEffect(() => {
    if (!mapRef.current) return;
    mapRef.current.getCanvas().style.cursor = isDrawing ? 'crosshair' : '';
  }, [isDrawing]);

  // Start Drawing
  const handleStartDrawing = () => {
    setIsDrawing(true);
    isDrawingRef.current = true;
    boundaryLockedRef.current = false;
    setDrawingPoints([]);
    drawingPointsRef.current = [];
    renderParcelBoundary([]);
  };

  // Finish Drawing - Locks boundary, fits camera smoothly, DOES NOT CLEAR POINTS!
  const handleFinishDrawing = () => {
    const pts = drawingPointsRef.current;
    if (pts.length < 3) {
      alert('Please click at least 3 points on the satellite map to define a closed parcel boundary.');
      return;
    }
    setIsDrawing(false);
    isDrawingRef.current = false;
    boundaryLockedRef.current = true;
    const closed = [...pts, pts[0]];
    const newPoly = {
      type: 'Polygon',
      coordinates: [closed],
    };

    // Keep points permanently visible on map
    renderParcelBoundary(pts, true);
    onParcelChange(newPoly);

    // Smoothly fit camera to the user's drawn parcel bounds
    const lngs = pts.map((p) => p[0]);
    const lats = pts.map((p) => p[1]);
    const minLng = Math.min(...lngs);
    const maxLng = Math.max(...lngs);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);

    if (mapRef.current) {
      mapRef.current.fitBounds(
        [[minLng, minLat], [maxLng, maxLat]],
        { padding: 80, maxZoom: 18, duration: 1200 }
      );
    }
  };

  // Undo Vertex
  const handleUndoVertex = () => {
    setDrawingPoints((prev) => {
      const next = prev.slice(0, -1);
      drawingPointsRef.current = next;
      renderParcelBoundary(next, false);
      return next;
    });
  };

  // Synchronized Survey & Parcel Clear
  const handleClearSurvey = useCallback(() => {
    setIsDrawing(false);
    isDrawingRef.current = false;
    boundaryLockedRef.current = false;
    setDrawingPoints([]);
    drawingPointsRef.current = [];

    // Trigger parent App state reset (parcelPolygon = null, analysisData = null, selectedPoint = null)
    if (onClearSurvey) {
      onClearSurvey();
    } else if (onParcelChange) {
      onParcelChange(null);
    }

    const map = mapRef.current;
    if (map) {
      // 1. Clears all drawn map vertices, polygon layers, and boundary geometry
      renderParcelBoundary([]);

      // 2. Resets area and perimeter measurements to zero
      const measSrc = map.getSource('parcel-measurements-src');
      if (measSrc) {
        measSrc.setData({ type: 'FeatureCollection', features: [] });
      }

      // 3. Removes all raster overlay layers (Quantum GWPI, SAR Moisture, InSAR Velocity)
      try {
        if (map.getLayer('active-raster-layer')) {
          map.removeLayer('active-raster-layer');
        }
        if (map.getSource('active-raster-src')) {
          map.removeSource('active-raster-src');
        }
      } catch (e) {
        console.warn('active-raster-src removal warning:', e);
      }

      // 4. Clears all vector overlays (drainage, lineaments, intersections crosshairs)
      try {
        if (map.getSource('drainage-source')) {
          map.getSource('drainage-source').setData({ type: 'FeatureCollection', features: [] });
        }
        if (map.getSource('lineaments-source')) {
          map.getSource('lineaments-source').setData({ type: 'FeatureCollection', features: [] });
        }
        if (map.getSource('intersections-source')) {
          map.getSource('intersections-source').setData({ type: 'FeatureCollection', features: [] });
        }
      } catch (e) {
        console.warn('vector sources clear warning:', e);
      }

      // 5. Clears all P1, P2, P3 borewell markers from the map
      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];

      // 6. Reset cursor
      const canvas = map.getCanvas();
      if (canvas) canvas.style.cursor = '';
    }
  }, [onClearSurvey, onParcelChange, renderParcelBoundary]);

  const handleClearParcel = handleClearSurvey;

  // Load Demo Farm Parcel
  const handleLoadDemo = (demoIndex = 0) => {
    const demo = DEMO_PARCELS[demoIndex] || DEMO_PARCELS[0];
    setIsDrawing(false);
    isDrawingRef.current = false;
    boundaryLockedRef.current = true;

    const coords = demo.polygon.coordinates[0];
    // Set drawing points so Points: N reflects the demo parcel
    const ptsNoDup = coords.slice(0, -1);
    setDrawingPoints(ptsNoDup);
    drawingPointsRef.current = ptsNoDup;
    onParcelChange(demo.polygon);

    if (mapRef.current) {
      mapRef.current.flyTo({
        center: demo.center,
        zoom: demo.zoom,
        pitch: 25,
        speed: 1.2,
      });

      renderParcelBoundary(coords, true);
    }
  };

  // Coordinate Search & Jump
  const handleSearchCoord = (e) => {
    e.preventDefault();
    const lat = parseFloat(searchLat);
    const lon = parseFloat(searchLon);
    if (isNaN(lat) || isNaN(lon)) return;

    if (mapRef.current) {
      mapRef.current.flyTo({
        center: [lon, lat],
        zoom: 15,
        speed: 1.4,
      });
    }
  };

  // Auto-Lock on "Run Hydrogeological Survey" click
  const handleRunSurvey = () => {
    let poly = parcelPolygon;

    // If user has marked >= 3 points without clicking 'Finish Boundary', auto-lock
    if (isDrawingRef.current && drawingPointsRef.current.length >= 3) {
      const pts = drawingPointsRef.current;
      const closed = [...pts, pts[0]];
      poly = {
        type: 'Polygon',
        coordinates: [closed],
      };
      setIsDrawing(false);
      isDrawingRef.current = false;
      boundaryLockedRef.current = true;
      renderParcelBoundary(pts, true);
      onParcelChange(poly);

      // Smoothly fit bounds
      const lngs = pts.map((p) => p[0]);
      const lats = pts.map((p) => p[1]);
      if (mapRef.current) {
        mapRef.current.fitBounds(
          [[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]],
          { padding: 80, maxZoom: 18, duration: 1000 }
        );
      }
    }

    onAnalyze(poly);
  };

  // Synchronize incoming parcelPolygon
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return;
    if (!parcelPolygon) {
      renderParcelBoundary([]);
      return;
    }
    const coords = parcelPolygon.coordinates?.[0];
    if (coords && coords.length >= 3) {
      renderParcelBoundary(coords, true);
    }
  }, [parcelPolygon, isMapLoaded, renderParcelBoundary]);

  // Dedicated Raster Overlay Switcher useEffect([analysisData, activeRaster, rasterOpacity])
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isMapLoaded) return;

    if (!analysisData) {
      try {
        if (map.getLayer('active-raster-layer')) {
          map.removeLayer('active-raster-layer');
        }
        if (map.getSource('active-raster-src')) {
          map.removeSource('active-raster-src');
        }
      } catch (err) {
        console.warn('active-raster-src cleanup warning:', err);
      }
      return;
    }

    const bounds = analysisData.bounds; // [west, south, east, north]
    if (!bounds || bounds.length !== 4) return;
    const [west, south, east, north] = bounds;

    // MapLibre GL 'image' source exact 4-corner format:
    // [[west, north], [east, north], [east, south], [west, south]]
    const imgCoordinates = [
      [west, north], // top-left
      [east, north], // top-right
      [east, south], // bottom-right
      [west, south], // bottom-left
    ];

    let currentUrl = null;
    if (activeRaster === 'gwpi') currentUrl = analysisData.gwpi_heatmap_base64;
    else if (activeRaster === 'sar') currentUrl = analysisData.sar_moisture_base64;
    else if (activeRaster === 'insar') currentUrl = analysisData.insar_heatmap_base64;

    try {
      const existingSource = map.getSource('active-raster-src');
      if (existingSource) {
        if (currentUrl && activeRaster !== 'none') {
          existingSource.updateImage({
            url: currentUrl,
            coordinates: imgCoordinates,
          });
          if (map.getLayer('active-raster-layer')) {
            map.setPaintProperty('active-raster-layer', 'raster-opacity', rasterOpacity);
          }
        } else {
          if (map.getLayer('active-raster-layer')) {
            map.setPaintProperty('active-raster-layer', 'raster-opacity', 0.0);
          }
        }
      } else if (currentUrl && activeRaster !== 'none') {
        map.addSource('active-raster-src', {
          type: 'image',
          url: currentUrl,
          coordinates: imgCoordinates,
        });

        // Add below parcel boundary
        const beforeLayer = map.getLayer('parcel-line') ? 'parcel-line' : undefined;
        map.addLayer(
          {
            id: 'active-raster-layer',
            type: 'raster',
            source: 'active-raster-src',
            paint: {
              'raster-opacity': rasterOpacity,
              'raster-fade-duration': 0,
            },
          },
          beforeLayer
        );
      }
    } catch (err) {
      console.warn('active-raster-src update error:', err);
    }
  }, [analysisData, activeRaster, rasterOpacity, isMapLoaded]);

  // Synchronize Vector Layers & P1/P2/P3 Markers
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isMapLoaded) return;

    if (!analysisData) {
      try {
        if (map.getSource('drainage-source')) {
          map.getSource('drainage-source').setData({ type: 'FeatureCollection', features: [] });
        }
        if (map.getSource('lineaments-source')) {
          map.getSource('lineaments-source').setData({ type: 'FeatureCollection', features: [] });
        }
        if (map.getSource('intersections-source')) {
          map.getSource('intersections-source').setData({ type: 'FeatureCollection', features: [] });
        }
        markersRef.current.forEach((m) => m.remove());
        markersRef.current = [];
      } catch (err) {
        console.warn('vector cleanup warning:', err);
      }
      return;
    }

    try {
      // 1. Update Drainage
      if (analysisData.drainage_geojson && map.getSource('drainage-source')) {
        map.getSource('drainage-source').setData(analysisData.drainage_geojson);
      }

      // 2. Update Lineaments
      if (analysisData.lineaments_geojson && map.getSource('lineaments-source')) {
        map.getSource('lineaments-source').setData(analysisData.lineaments_geojson);
      }

      // 3. Update Intersections
      if (analysisData.intersections_geojson && map.getSource('intersections-source')) {
        map.getSource('intersections-source').setData(analysisData.intersections_geojson);
      }

      // 4. Render Glowing Geologist P1, P2, P3 Target Pins
      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];

      const points = analysisData.borewell_points || [];
      points.forEach((pt) => {
        const el = document.createElement('div');
        el.className = 'group relative flex items-center justify-center cursor-pointer';

        const pulseRing = document.createElement('div');
        pulseRing.className =
          'absolute w-9 h-9 rounded-full bg-[#00FF66]/25 border border-[#00FF66] radar-marker-pulse';
        el.appendChild(pulseRing);

        const pin = document.createElement('div');
        pin.className =
          'relative z-10 w-7 h-7 rounded-full bg-[#00FF66] text-[#0a0f18] font-black text-xs flex items-center justify-center shadow-[0_0_15px_#00FF66] border-2 border-white transition-transform group-hover:scale-125';
        pin.innerText = pt.id;
        el.appendChild(pin);

        const marker = new maplibregl.Marker({ element: el, anchor: 'center' })
          .setLngLat([pt.lon, pt.lat])
          .addTo(map);

        el.addEventListener('click', (ev) => {
          ev.stopPropagation();
          onSelectPoint(pt);

          if (popupRef.current) popupRef.current.remove();
          popupRef.current = new maplibregl.Popup({ offset: 16, maxWidth: '340px' })
            .setLngLat([pt.lon, pt.lat])
            .setHTML(`
              <div class="text-xs space-y-2 p-1 font-sans">
                <div class="flex items-center justify-between pb-1 border-b border-slate-700">
                  <span class="font-extrabold text-[#00FF66] text-sm tracking-wide">Target ${pt.id}</span>
                  <span class="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 font-mono text-[10px] font-bold border border-emerald-500/40">
                    GWPI ${pt.gwpi_score.toFixed(1)} / 100
                  </span>
                </div>
                <div class="grid grid-cols-2 gap-2 text-[11px] font-mono py-1">
                  <div><span class="text-slate-400">Depth:</span> <b class="text-amber-400">${pt.recommended_depth_m.toFixed(1)}m</b> (${Math.round(pt.recommended_depth_m * 3.28084)}ft)</div>
                  <div><span class="text-slate-400">Est. Yield:</span> <b class="text-cyan-400">${Math.round(pt.estimated_yield_lph).toLocaleString()}</b> LPH</div>
                  <div><span class="text-slate-400">Azimuth:</span> <b>${pt.fracture_azimuth_deg.toFixed(1)}°</b></div>
                  <div><span class="text-slate-400">Stream:</span> <b>Order ${pt.stream_order}</b></div>
                </div>
                <div class="text-[11px] text-slate-300 bg-slate-900/80 p-2 rounded border border-slate-800 leading-snug">
                  ${pt.reasoning}
                </div>
              </div>
            `)
            .addTo(map);
        });

        markersRef.current.push(marker);
      });
    } catch (err) {
      console.warn('analysisData synchronization warning:', err);
    }
  }, [analysisData, isMapLoaded]);

  // Synchronize Layer Visibilities
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isMapLoaded) return;

    try {
      if (map.getLayer('drainage-lines')) {
        map.setLayoutProperty('drainage-lines', 'visibility', showDrainage ? 'visible' : 'none');
        map.setLayoutProperty('drainage-casing', 'visibility', showDrainage ? 'visible' : 'none');
      }
      if (map.getLayer('lineaments-lines')) {
        map.setLayoutProperty('lineaments-lines', 'visibility', showLineaments ? 'visible' : 'none');
      }
      if (map.getLayer('intersections-ring')) {
        map.setLayoutProperty('intersections-ring', 'visibility', showIntersections ? 'visible' : 'none');
        map.setLayoutProperty('intersections-center', 'visibility', showIntersections ? 'visible' : 'none');
      }
    } catch (err) {
      console.warn('layer visibility sync warning:', err);
    }
  }, [showDrainage, showLineaments, showIntersections, isMapLoaded]);

  // When selectedPoint changes externally, pan map to it
  useEffect(() => {
    if (!mapRef.current || !selectedPoint) return;
    mapRef.current.flyTo({
      center: [selectedPoint.lon, selectedPoint.lat],
      zoom: 16.5,
      speed: 1.5,
      pitch: 30,
    });
  }, [selectedPoint]);

  return (
    <div className="relative w-full h-full bg-[#0a0f18] overflow-hidden">
      {/* Map Canvas */}
      <div ref={mapContainerRef} className="w-full h-full" />

      {/* Top Floating Control Bar (Pointer-events-none on wrapper) */}
      <div className="absolute top-4 left-4 right-4 z-20 flex flex-wrap items-center justify-between gap-3 pointer-events-none">
        {/* Left: GPS Search & Village Jump */}
        <form
          onSubmit={handleSearchCoord}
          className="pointer-events-auto flex items-center gap-1.5 bg-[#0e1626]/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700/80 shadow-2xl"
        >
          <Search className="w-4 h-4 text-cyan-400 shrink-0" />
          <input
            type="text"
            value={searchLat}
            onChange={(e) => setSearchLat(e.target.value)}
            placeholder="Lat (e.g. 13.435)"
            className="w-24 bg-transparent text-xs text-white placeholder-slate-400 focus:outline-none font-mono"
          />
          <span className="text-slate-500 text-xs">,</span>
          <input
            type="text"
            value={searchLon}
            onChange={(e) => setSearchLon(e.target.value)}
            placeholder="Lon (e.g. 77.727)"
            className="w-24 bg-transparent text-xs text-white placeholder-slate-400 focus:outline-none font-mono"
          />
          <button
            type="submit"
            className="px-2 py-1 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-[11px] font-semibold transition"
          >
            Fly
          </button>
        </form>

        {/* Center: Parcel Boundary Drawing Tools */}
        <div className="pointer-events-auto flex items-center gap-2 bg-[#0e1626]/90 backdrop-blur-md p-1.5 rounded-lg border border-slate-700/80 shadow-2xl">
          {!isDrawing ? (
            <div className="flex items-center gap-1.5">
              <button
                onClick={handleStartDrawing}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 rounded text-xs font-semibold transition"
              >
                <Crosshair className="w-3.5 h-3.5" />
                Draw Parcel Boundary
              </button>
              {(parcelPolygon || analysisData || drawingPoints.length > 0) && (
                <button
                  onClick={handleClearSurvey}
                  className="flex items-center gap-1 px-2.5 py-1.5 bg-rose-900/30 hover:bg-rose-900/50 text-rose-300 border border-rose-800/60 rounded text-xs font-medium transition"
                  title="Clear Survey & Reset Workspace"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear
                </button>
              )}
            </div>
          ) : (
            <>
              <span className="text-[11px] text-amber-300 font-mono px-2 animate-pulse flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                Points: {drawingPoints.length}
              </span>
              {drawingPoints.length > 0 && (
                <button
                  onClick={handleUndoVertex}
                  className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs font-medium transition"
                  title="Undo last vertex"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              )}
              <button
                onClick={handleFinishDrawing}
                disabled={drawingPoints.length < 3}
                className="flex items-center gap-1 px-3 py-1 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded text-xs font-bold transition shadow-sm"
              >
                <CheckCircle className="w-3.5 h-3.5" />
                Finish Boundary
              </button>
              <button
                onClick={handleClearSurvey}
                className="p-1.5 bg-rose-900/30 hover:bg-rose-900/50 text-rose-300 rounded text-xs transition"
                title="Cancel Drawing"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </>
          )}

          {/* Load Demo Farm */}
          <div className="flex items-center gap-1.5 border-l border-slate-700/80 pl-2">
            <button
              onClick={() => handleLoadDemo(0)}
              className="flex items-center gap-1 px-2.5 py-1.5 bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-600/50 rounded text-xs font-medium transition"
              title="Karnataka Granite-Gneiss Demo Farm"
            >
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              Load Demo Farm (IN)
            </button>
          </div>
        </div>

        {/* Right: Main Analyze Pipeline Trigger */}
        <div className="pointer-events-auto">
          <button
            onClick={handleRunSurvey}
            disabled={(!parcelPolygon && drawingPoints.length < 3) || isLoading}
            className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-40 text-white rounded-lg font-bold text-xs tracking-wide shadow-[0_0_20px_rgba(0,229,255,0.4)] transition transform active:scale-95"
          >
            {isLoading ? (
              <>
                <Activity className="w-4 h-4 animate-spin" />
                Analyzing Hydrogeology...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                Run Hydrogeological Survey
              </>
            )}
          </button>
        </div>
      </div>

      {/* Top Bar Measurement HUD Pill */}
      {currentMetrics && currentMetrics.area_acres > 0 && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-20 pointer-events-none transition-all">
          <div className="pointer-events-auto flex items-center gap-2.5 px-4 py-1.5 rounded-full bg-[#0e1626]/95 border border-amber-500/60 shadow-[0_0_20px_rgba(245,158,11,0.3)] backdrop-blur-md text-xs font-mono">
            <span className="text-amber-400 font-black flex items-center gap-1.5">
              <span>📐</span>
              <span>{currentMetrics.area_acres.toFixed(2)} Acres</span>
            </span>
            <span className="text-amber-200/90 font-semibold">({currentMetrics.acres_cents_str})</span>
            <span className="text-slate-600 font-bold">•</span>
            <span className="text-cyan-300 font-medium">{currentMetrics.area_hectares.toFixed(2)} ha</span>
            <span className="text-slate-600 font-bold">•</span>
            <span className="text-emerald-400 font-medium">
              Perimeter: {currentMetrics.perimeter_m.toLocaleString()} m ({currentMetrics.perimeter_ft.toLocaleString()} ft)
            </span>
          </div>
        </div>
      )}

      {/* Floating Left Layer Controls & Overlays */}
      <div className="absolute bottom-6 left-4 z-20 w-72 bg-[#0e1626]/95 backdrop-blur-md border border-slate-700/80 rounded-xl shadow-2xl overflow-hidden font-sans pointer-events-auto">
        <div
          onClick={() => setShowLayerMenu(!showLayerMenu)}
          className="flex items-center justify-between px-3.5 py-2.5 bg-slate-900/80 border-b border-slate-700/60 cursor-pointer hover:bg-slate-800/60 transition"
        >
          <div className="flex items-center gap-2 text-xs font-bold text-slate-200">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Hydrogeological Layer Controls</span>
          </div>
          <ChevronDown
            className={`w-4 h-4 text-slate-400 transition-transform ${showLayerMenu ? 'rotate-180' : ''}`}
          />
        </div>

        {showLayerMenu && (
          <div className="p-3.5 space-y-3.5 text-xs text-slate-300">
            {/* Vector Layers */}
            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-bold mb-2">
                Structural & Drainage Vectors
              </div>
              <div className="space-y-1.5">
                <label className="flex items-center justify-between cursor-pointer hover:text-white">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-0.5 bg-[#00E5FF] rounded"></span>
                    <span>Micro-Drainage (Strahler 1-4)</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={showDrainage}
                    onChange={(e) => setShowDrainage(e.target.checked)}
                    className="accent-cyan-400 rounded cursor-pointer"
                  />
                </label>

                <label className="flex items-center justify-between cursor-pointer hover:text-white">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-0.5 bg-[#00FFCC] border-b border-dashed border-yellow-300"></span>
                    <span>Fracture Lineaments</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={showLineaments}
                    onChange={(e) => setShowLineaments(e.target.checked)}
                    className="accent-cyan-400 rounded cursor-pointer"
                  />
                </label>

                <label className="flex items-center justify-between cursor-pointer hover:text-white">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full border border-cyan-400"></span>
                    <span>Drainage-Fracture Intersections</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={showIntersections}
                    onChange={(e) => setShowIntersections(e.target.checked)}
                    className="accent-cyan-400 rounded cursor-pointer"
                  />
                </label>
              </div>
            </div>

            {/* Raster Satellite Overlays */}
            <div className="pt-2 border-t border-slate-700/60">
              <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-bold mb-2">
                Satellite & Quantum Raster Overlays
              </div>
              <div className="grid grid-cols-2 gap-1.5 mb-2.5">
                <button
                  onClick={() => setActiveRaster('gwpi')}
                  className={`py-1.5 px-2 rounded text-[11px] font-semibold border transition text-center ${
                    activeRaster === 'gwpi'
                      ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-sm'
                      : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-white'
                  }`}
                >
                  Quantum GWPI
                </button>
                <button
                  onClick={() => setActiveRaster('sar')}
                  className={`py-1.5 px-2 rounded text-[11px] font-semibold border transition text-center ${
                    activeRaster === 'sar'
                      ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-sm'
                      : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-white'
                  }`}
                >
                  SAR Moisture
                </button>
                <button
                  onClick={() => setActiveRaster('insar')}
                  className={`py-1.5 px-2 rounded text-[11px] font-semibold border transition text-center ${
                    activeRaster === 'insar'
                      ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-sm'
                      : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-white'
                  }`}
                >
                  InSAR Velocity
                </button>
                <button
                  onClick={() => setActiveRaster('none')}
                  className={`py-1.5 px-2 rounded text-[11px] font-semibold border transition text-center ${
                    activeRaster === 'none'
                      ? 'bg-slate-700 border-slate-500 text-white'
                      : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-white'
                  }`}
                >
                  None (Satellite)
                </button>
              </div>

              {/* Opacity Slider */}
              {activeRaster !== 'none' && (
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                    <span>Raster Opacity</span>
                    <span>{Math.round(rasterOpacity * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={rasterOpacity}
                    onChange={(e) => setRasterOpacity(parseFloat(e.target.value))}
                    className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Dynamic Floating Vertical Legend Colorbars */}
      {activeRaster === 'gwpi' && (
        <div className="absolute top-20 right-4 z-20 flex flex-col items-center bg-[#0e1626]/90 backdrop-blur-md p-2.5 rounded-xl border border-slate-700 shadow-2xl font-mono text-[10px] pointer-events-auto">
          <div className="font-bold text-slate-200 mb-1">Quantum GWPI</div>
          <div className="text-cyan-400">100 (Deep Potential)</div>
          <div className="w-4 h-32 my-1 rounded-sm bg-gradient-to-b from-[#0e50c8] via-[#00e5ff] via-33%-[#facc15] to-[#dc2626] shadow-inner"></div>
          <div className="text-yellow-400">50 (Moderate)</div>
          <div className="text-rose-400">0 (Very Low)</div>
        </div>
      )}

      {activeRaster === 'sar' && (
        <div className="absolute top-20 right-4 z-20 flex flex-col items-center bg-[#0e1626]/90 backdrop-blur-md p-2.5 rounded-xl border border-slate-700 shadow-2xl font-mono text-[10px] pointer-events-auto">
          <div className="font-bold text-slate-200 mb-1">SAR Moisture (dB)</div>
          <div className="text-cyan-400">-5 dB (Dielectric Peak)</div>
          <div className="w-4 h-32 my-1 rounded-sm bg-gradient-to-b from-[#00e5ff] via-[#e2e8f0] to-[#1e293b] shadow-inner"></div>
          <div className="text-slate-300">-15 dB (Normal)</div>
          <div className="text-slate-500">-25 dB (Dry Soil)</div>
        </div>
      )}

      {activeRaster === 'insar' && (
        <div className="absolute top-20 right-4 z-20 flex flex-col items-center bg-[#0e1626]/90 backdrop-blur-md p-2.5 rounded-xl border border-slate-700 shadow-2xl font-mono text-[10px] pointer-events-auto">
          <div className="font-bold text-slate-200 mb-1">InSAR Velocity</div>
          <div className="text-rose-400">+10 mm/yr (Uplift)</div>
          <div className="w-4 h-32 my-1 rounded-sm bg-gradient-to-b from-[#eb1e1e] via-[#fae628] via-[#00e5d2] to-[#0e50c8] shadow-inner"></div>
          <div className="text-yellow-300">0 mm/yr (Stable)</div>
          <div className="text-cyan-400">-10 mm/yr (Subsidence)</div>
        </div>
      )}

      {/* Drawing Instructions Toast */}
      {isDrawing && (
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 bg-amber-500/90 text-black font-semibold text-xs px-4 py-2 rounded-full shadow-2xl animate-bounce flex items-center gap-2 pointer-events-none select-none">
          <Crosshair className="w-4 h-4" />
          Click vertices on the satellite map around your parcel boundary, then click "Finish Boundary".
        </div>
      )}
    </div>
  );
}
