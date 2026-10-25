import React, { useState, Component } from 'react';
import Header from './components/Header';
import GeologistMap from './components/GeologistMap';
import BorewellReport from './components/BorewellReport';
import { DEMO_PARCELS } from './data/demoParcels';
import { PanelRightClose, PanelRightOpen, AlertCircle, RefreshCw } from 'lucide-react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full flex flex-col items-center justify-center p-6 bg-[#0a0f18] text-center font-sans">
          <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/80 max-w-md space-y-3">
            <div className="flex items-center gap-2 text-rose-400 font-bold text-sm">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <span>Map Workspace Display Error</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              {this.state.error?.message || 'A transient WebGL or map stylesheet error occurred.'}
            </p>
            <button
              onClick={this.handleReset}
              className="px-3 py-1.5 bg-rose-700 hover:bg-rose-600 text-white rounded text-xs font-semibold flex items-center gap-1.5 mx-auto transition"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Reload Workspace
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default function App() {
  const [parcelPolygon, setParcelPolygon] = useState(DEMO_PARCELS[0].polygon);
  const [bufferM, setBufferM] = useState(1500.0);
  const [analysisData, setAnalysisData] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [errorMessage, setErrorMessage] = useState(null);

  // Synchronized unified survey clear handler
  const handleClearSurvey = () => {
    setParcelPolygon(null);
    setAnalysisData(null);
    setSelectedPoint(null);
    setErrorMessage(null);
  };

  // Run full hydrogeological analysis pipeline
  const handleAnalyze = async (overridePoly = null) => {
    const targetPoly = overridePoly || parcelPolygon;
    if (!targetPoly) {
      alert('Please draw or select a parcel boundary first.');
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const response = await fetch('/api/analyze-parcel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parcel_geometry: targetPoly,
          buffer_m: bufferM,
        }),
      });

      if (!response.ok) {
        const errDetail = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errDetail.detail || `Server responded with ${response.status}`);
      }

      const data = await response.json();
      setAnalysisData(data);
      if (data.borewell_points && data.borewell_points.length > 0) {
        setSelectedPoint(data.borewell_points[0]); // Default to P1
      }
      setSidebarOpen(true);
    } catch (err) {
      console.error('Analysis error:', err);
      setErrorMessage(err.message || 'Hydrogeological analysis pipeline encountered an error.');
    } finally {
      setIsLoading(false);
    }
  };

  // Export Google Earth KML
  const handleExportKml = async () => {
    if (!analysisData && !parcelPolygon) return;
    setIsExporting(true);

    try {
      // Send either existing analysis or parcel request
      const payload = analysisData
        ? analysisData
        : {
            parcel_geometry: parcelPolygon,
            buffer_m: bufferM,
          };

      const response = await fetch('/api/export-kml', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`KML export failed: ${response.statusText}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'BoreSight_Q_Hydro_Report.kml';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('KML export error:', err);
      alert(`Could not export KML: ${err.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#0a0f18] text-slate-100 overflow-hidden font-sans">
      {/* Top Header */}
      <Header bufferM={bufferM} onBufferChange={setBufferM} />

      {/* Error Banner */}
      {errorMessage && (
        <div className="bg-rose-950/90 border-b border-rose-800 text-rose-200 px-4 py-2 text-xs flex items-center justify-between z-40">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-xs text-rose-300 hover:text-white underline ml-4"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Workspace: Map on Left, Sidebar on Right */}
      <div className="flex-1 flex relative overflow-hidden">
        {/* Map Workspace with Error Boundary */}
        <div className="flex-1 relative h-full">
          <ErrorBoundary>
            <GeologistMap
              analysisData={analysisData}
              isLoading={isLoading}
              onAnalyze={handleAnalyze}
              selectedPoint={selectedPoint}
              onSelectPoint={setSelectedPoint}
              parcelPolygon={parcelPolygon}
              onParcelChange={setParcelPolygon}
              onClearSurvey={handleClearSurvey}
            />
          </ErrorBoundary>

          {/* Toggle Sidebar Collapse Button */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="absolute top-4 right-4 z-20 p-2 rounded-lg bg-[#0e1626]/90 backdrop-blur-md border border-slate-700 text-slate-300 hover:text-white shadow-xl transition"
            title={sidebarOpen ? 'Collapse Consultation Sidebar' : 'Expand Consultation Sidebar'}
          >
            {sidebarOpen ? (
              <PanelRightClose className="w-4 h-4" />
            ) : (
              <PanelRightOpen className="w-4 h-4 text-cyan-400" />
            )}
          </button>
        </div>

        {/* Right Consultation Sidebar */}
        <div
          className={`h-full transition-all duration-300 ease-in-out relative z-30 ${
            sidebarOpen ? 'w-[430px] opacity-100' : 'w-0 opacity-0 overflow-hidden'
          }`}
        >
          <BorewellReport
            analysisData={analysisData}
            parcelPolygon={parcelPolygon}
            selectedPoint={selectedPoint}
            onSelectPoint={setSelectedPoint}
            isExporting={isExporting}
            onExportKml={handleExportKml}
          />
        </div>
      </div>
    </div>
  );
}
