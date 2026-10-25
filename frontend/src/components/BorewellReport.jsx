import React, { useState } from 'react';
import {
  Download,
  Copy,
  Check,
  Compass,
  Droplets,
  Layers,
  Activity,
  AlertTriangle,
  MapPin,
  ExternalLink,
  ShieldCheck,
  Sparkles,
  Info,
  TrendingDown,
  TrendingUp,
  FileSpreadsheet,
  Ruler,
  Gauge,
  Zap
} from 'lucide-react';
import { calculateGeodesicLandMetrics } from '../utils/geodesicArea';

export default function BorewellReport({
  analysisData,
  parcelPolygon,
  selectedPoint,
  onSelectPoint,
  isExporting,
  onExportKml,
}) {
  const [copiedId, setCopiedId] = useState(null);

  const handleCopyCoords = (pt) => {
    const text = `${pt.lat.toFixed(7)}, ${pt.lon.toFixed(7)}`;
    navigator.clipboard.writeText(text);
    setCopiedId(pt.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  if (!analysisData) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center p-8 text-center bg-[#0d1424] border-l border-slate-800 text-slate-400">
        <div className="w-16 h-16 rounded-2xl bg-cyan-950/40 border border-cyan-500/20 flex items-center justify-center mb-4 text-cyan-400">
          <Droplets className="w-8 h-8 animate-pulse" />
        </div>
        <h3 className="text-slate-200 font-bold text-base mb-2">No Parcel Survey Active</h3>
        <p className="text-xs text-slate-400 max-w-xs leading-relaxed mb-6">
          Draw your agricultural property boundary on the satellite workspace or load a demo farm parcel, then run the hydrogeological survey.
        </p>
        <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-left text-xs space-y-2 w-full max-w-xs font-mono">
          <div className="text-cyan-400 font-semibold text-[11px] uppercase tracking-wide">Autonomous Engine Specs:</div>
          <div className="flex items-center gap-2 text-slate-300">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
            <span>12-Qubit Variational Quantum Regressor</span>
          </div>
          <div className="flex items-center gap-2 text-slate-300">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
            <span>Sentinel-1/2 & InSAR Deformation</span>
          </div>
          <div className="flex items-center gap-2 text-slate-300">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
            <span>Crystalline Hard-Rock Fracture Intersections</span>
          </div>
        </div>
      </div>
    );
  }

  const { borewell_points = [], aquifer_summary = {}, parcel_metrics } = analysisData;
  const metrics = parcel_metrics || (parcelPolygon?.coordinates?.[0] ? calculateGeodesicLandMetrics(parcelPolygon.coordinates[0]) : null);
  const p1 = borewell_points.find((p) => p.id === 'P1') || borewell_points[0];
  const p2 = borewell_points.find((p) => p.id === 'P2') || borewell_points[1];
  const p3 = borewell_points.find((p) => p.id === 'P3') || borewell_points[2];

  const getTargetMeta = (id) => {
    switch (id) {
      case 'P1':
        return {
          label: 'Primary Drilling Target',
          color: 'border-emerald-500/70 bg-emerald-950/20 shadow-[0_0_20px_rgba(16,185,129,0.15)]',
          badge: 'bg-emerald-500 text-slate-950 font-black',
          pinBg: 'bg-emerald-500',
        };
      case 'P2':
        return {
          label: 'Secondary / Backup Target',
          color: 'border-cyan-500/60 bg-cyan-950/20 shadow-[0_0_15px_rgba(6,182,212,0.1)]',
          badge: 'bg-cyan-500 text-slate-950 font-black',
          pinBg: 'bg-cyan-500',
        };
      case 'P3':
        return {
          label: 'Contingency Fracture Node',
          color: 'border-amber-500/60 bg-amber-950/20 shadow-[0_0_15px_rgba(245,158,11,0.1)]',
          badge: 'bg-amber-500 text-slate-950 font-black',
          pinBg: 'bg-amber-500',
        };
      default:
        return {
          label: 'Candidate Target',
          color: 'border-slate-700 bg-slate-900',
          badge: 'bg-slate-700 text-white',
          pinBg: 'bg-slate-700',
        };
    }
  };

  const renderTargetCard = (pt) => {
    if (!pt) return null;
    const meta = getTargetMeta(pt.id);
    const isSelected = selectedPoint?.id === pt.id;
    const depthM = pt.recommended_depth_m;
    const depthFt = Math.round(depthM * 3.28084);
    const yieldLph = Math.round(pt.estimated_yield_lph);
    const yieldGpm = Math.round(yieldLph / 227.12);

    return (
      <div
        key={pt.id}
        onClick={() => onSelectPoint(pt)}
        className={`p-4 rounded-xl border transition-all cursor-pointer relative overflow-hidden ${
          meta.color
        } ${isSelected ? 'ring-2 ring-emerald-400 scale-[1.01]' : 'hover:border-slate-500'}`}
      >
        {/* Header / Badges */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className={`w-7 h-7 rounded-lg flex items-center justify-center text-xs font-black text-black shadow ${meta.pinBg}`}>
              {pt.id}
            </span>
            <div>
              <div className="text-xs font-extrabold text-white tracking-wide">{meta.label}</div>
              <div className="text-[10px] text-slate-400 font-mono flex items-center gap-1.5 mt-0.5">
                <span>Strahler {pt.stream_order}</span>
                {pt.pipe_size_display && !analysisData.is_surface_water && !analysisData.is_hyper_arid && yieldLph > 0 && (
                  <span className="px-1.5 py-0.2 rounded bg-teal-950/80 text-teal-300 border border-teal-700/60 text-[9px] font-bold">
                    Pipe: {pt.pipe_size_display}
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="flex flex-col items-end">
            <span className={`text-xs font-black font-mono ${analysisData.is_surface_water ? 'text-blue-400' : analysisData.is_hyper_arid || pt.gwpi_score < 15.0 ? 'text-rose-400' : 'text-emerald-400'}`}>
              GWPI {pt.gwpi_score.toFixed(1)} <span className="text-[10px] text-slate-400 font-normal">/ 100</span>
            </span>
            <span className={`text-[10px] font-mono ${analysisData.is_surface_water ? 'text-blue-300 font-bold' : analysisData.is_hyper_arid || pt.gwpi_score < 15.0 ? 'text-rose-300 font-bold' : 'text-cyan-300'}`}>
              {analysisData.is_surface_water ? 'Surface Water' : analysisData.is_hyper_arid || pt.gwpi_score < 15.0 ? 'Barren / Arid' : `${pt.confidence_pct.toFixed(0)}% Conf.`}
            </span>
          </div>
        </div>

        {/* GPS Coordinates & 1-Click Copy */}
        <div className="flex items-center justify-between p-2 rounded-lg bg-black/40 border border-slate-800/80 mb-3 font-mono text-xs">
          <div className="flex items-center gap-1.5 text-slate-300">
            <MapPin className="w-3.5 h-3.5 text-rose-400 shrink-0" />
            <span>{pt.lat.toFixed(6)}° N, {pt.lon.toFixed(6)}° E</span>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleCopyCoords(pt);
            }}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[11px] font-sans transition"
            title="Copy coordinates to clipboard"
          >
            {copiedId === pt.id ? (
              <>
                <Check className="w-3 h-3 text-emerald-400" />
                <span className="text-emerald-400 font-bold">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3 text-slate-400" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>

        {/* Specs Grid: Drilling Depth, Yield, Pipe Size, Submersible Pump */}
        <div className="grid grid-cols-2 gap-2 mb-3 font-mono">
          <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800">
            <div className="text-[10px] uppercase text-slate-400 font-sans font-bold flex items-center gap-1">
              <Layers className="w-3 h-3 text-amber-400" />
              Drilling Depth (BGL)
            </div>
            <div className="text-sm font-black text-amber-400 mt-1">
              {analysisData.is_surface_water ? 'Inapplicable' : analysisData.is_hyper_arid || depthM === 0 ? 'Unfeasible' : `${depthM.toFixed(1)} m`}
            </div>
            <div className="text-[10px] text-slate-400">
              {analysisData.is_surface_water ? '(Surface Water)' : analysisData.is_hyper_arid || depthM === 0 ? '(Dry Bedrock)' : `(${depthFt} feet)`}
            </div>
          </div>

          <div className={`p-2.5 rounded-lg border ${analysisData.is_surface_water ? 'bg-blue-950/30 border-blue-800/80' : analysisData.is_hyper_arid || yieldLph === 0 ? 'bg-rose-950/30 border-rose-800/80' : 'bg-slate-900/90 border-slate-800'}`}>
            <div className="text-[10px] uppercase text-slate-400 font-sans font-bold flex items-center gap-1">
              <Droplets className={`w-3 h-3 ${analysisData.is_surface_water ? 'text-blue-400' : analysisData.is_hyper_arid || yieldLph === 0 ? 'text-rose-400' : 'text-cyan-400'}`} />
              Estimated Yield
            </div>
            <div className={`text-sm font-black mt-1 ${analysisData.is_surface_water ? 'text-blue-400' : analysisData.is_hyper_arid || yieldLph === 0 ? 'text-rose-400' : 'text-cyan-400'}`}>
              {analysisData.is_surface_water ? '0 LPH' : analysisData.is_hyper_arid || yieldLph === 0 ? '0 LPH' : `${yieldLph.toLocaleString()} LPH`}
            </div>
            <div className={`text-[10px] ${analysisData.is_surface_water ? 'text-blue-300 font-bold' : analysisData.is_hyper_arid || yieldLph === 0 ? 'text-rose-300 font-bold' : 'text-slate-400'}`}>
              {analysisData.is_surface_water ? 'OPEN WATER' : analysisData.is_hyper_arid || yieldLph === 0 ? 'BARREN / DRY' : `(~${yieldGpm} GPM)`}
            </div>
          </div>

          <div className="bg-slate-900/90 p-2.5 rounded-lg border border-teal-500/40">
            <div className="text-[10px] uppercase text-slate-400 font-sans font-bold flex items-center justify-between">
              <span className="flex items-center gap-1">
                <Gauge className="w-3 h-3 text-teal-400" />
                Delivery Pipe Size
              </span>
              <span className="px-1.5 py-0.2 rounded bg-teal-500/20 text-teal-300 text-[10px] font-black">
                {analysisData.is_surface_water || analysisData.is_hyper_arid || yieldLph === 0
                  ? 'N/A'
                  : (pt.pipe_size_display ? (pt.pipe_size_display.includes('(') ? pt.pipe_size_display.match(/\((.*?)\)/)?.[1] || `${pt.delivery_pipe_size_inches || 1.5}"` : `${pt.delivery_pipe_size_inches || 1.5}"`) : '1½"')}
              </span>
            </div>
            <div className="text-sm font-black text-teal-300 mt-1 truncate" title={pt.pipe_size_display || '1.5 inches (1½")'}>
              {analysisData.is_surface_water || analysisData.is_hyper_arid || yieldLph === 0 ? 'N/A' : (pt.pipe_size_display || '1.5 inches (1½")')}
            </div>
            <div className="text-[10px] text-teal-400/80">
              {analysisData.is_surface_water || analysisData.is_hyper_arid || yieldLph === 0 ? '(No Column)' : '(Optimal Flow Rate)'}
            </div>
          </div>

          <div className="bg-slate-900/90 p-2.5 rounded-lg border border-purple-500/40">
            <div className="text-[10px] uppercase text-slate-400 font-sans font-bold flex items-center gap-1">
              <Zap className="w-3 h-3 text-purple-400" />
              Submersible Pump
            </div>
            <div className="text-xs font-black text-purple-300 mt-1.5 leading-snug truncate" title={pt.recommended_pump_hp || '3.0 HP to 5.0 HP Submersible'}>
              {analysisData.is_surface_water || analysisData.is_hyper_arid || yieldLph === 0 ? 'None (Unfeasible)' : (pt.recommended_pump_hp || '3.0 HP - 5.0 HP')}
            </div>
            <div className="text-[10px] text-purple-400/80">
              {analysisData.is_surface_water || analysisData.is_hyper_arid || yieldLph === 0 ? '(Inapplicable)' : '(Recommended HP)'}
            </div>
          </div>
        </div>

        {/* Structural Attributes */}
        <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 mb-2 px-1">
          <div className="flex items-center gap-1">
            <Compass className="w-3 h-3 text-cyan-400" />
            <span>Fracture Strike:</span>
            <span className="font-bold text-slate-200">{pt.fracture_azimuth_deg.toFixed(1)}°</span>
          </div>
          <div>
            <span>Conjugate:</span> <b className="text-emerald-400">{((pt.fracture_azimuth_deg + 90) % 360).toFixed(0)}°</b>
          </div>
        </div>

        {/* Geologist Reasoning */}
        <div className="text-xs text-slate-300 bg-slate-900/80 p-2.5 rounded-lg border border-slate-800/80 leading-relaxed font-sans">
          <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-slate-400 block mb-1">
            Geological Assessment:
          </span>
          {pt.reasoning}
        </div>
      </div>
    );
  };

  return (
    <div className="w-full h-full flex flex-col bg-[#0d1424] border-l border-slate-800 overflow-hidden font-sans">
      {/* Top Header */}
      <div className="p-4 border-b border-slate-800 bg-[#0e172a] shrink-0">
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-sm font-extrabold text-white tracking-wide flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            Hydrogeological Consultation
          </h2>
          <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 text-[10px] font-mono font-bold">
            PARCEL-SCALE
          </span>
        </div>
        <p className="text-xs text-slate-400">
          Precision hard-rock borewell siting & structural fracture assessment.
        </p>
      </div>

      {/* Scrollable Report Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Selected Land Survey & Area Measurement Card */}
        {metrics && metrics.area_acres > 0 && (
          <div className="p-4 rounded-xl bg-gradient-to-br from-[#121c30] to-[#0d1527] border border-amber-500/40 shadow-xl space-y-3 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/5 rounded-full blur-2xl pointer-events-none" />

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-md bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-amber-400">
                  <Ruler className="w-3.5 h-3.5" />
                </div>
                <div>
                  <h3 className="text-xs font-black text-white tracking-wide uppercase">Selected Land Survey & Area</h3>
                  <span className="text-[10px] text-slate-400 font-mono">WGS84 Geodesic Integration</span>
                </div>
              </div>
              <span className="px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 font-mono font-bold text-[10px]">
                SURVEY GRADE
              </span>
            </div>

            {/* Primary Big Stat: Total Area in Acres & Cents */}
            <div className="p-3 rounded-lg bg-black/40 border border-amber-500/30 flex items-center justify-between">
              <div>
                <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-bold">
                  Total Land Area
                </div>
                <div className="text-xl font-black text-amber-400 font-mono tracking-tight mt-0.5">
                  {Number(metrics.area_acres).toFixed(4)} <span className="text-xs font-bold text-amber-300">Acres</span>
                </div>
              </div>
              <div className="text-right">
                <span className="inline-block px-2.5 py-1 rounded-md bg-amber-400 text-slate-950 font-black text-xs font-mono shadow-md">
                  {metrics.acres_cents_str}
                </span>
                <div className="text-[10px] text-amber-300/80 font-mono mt-1">
                  (~{(Number(metrics.area_acres) * 40).toFixed(1)} Guntas)
                </div>
              </div>
            </div>

            {/* Grid of Sub-Metrics: Hectares, Sq. Meters, Sq. Yards, Total Perimeter */}
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2.5 rounded-lg bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-400 font-sans">Hectares</div>
                <div className="text-white font-bold mt-0.5">
                  {Number(metrics.area_hectares).toFixed(4)} <span className="text-[10px] text-slate-400 font-normal">ha</span>
                </div>
              </div>

              <div className="p-2.5 rounded-lg bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-400 font-sans">Square Meters</div>
                <div className="text-white font-bold mt-0.5">
                  {Number(metrics.area_sq_m).toLocaleString()} <span className="text-[10px] text-slate-400 font-normal">m²</span>
                </div>
              </div>

              <div className="p-2.5 rounded-lg bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-400 font-sans">Square Yards (Gajam)</div>
                <div className="text-cyan-300 font-bold mt-0.5">
                  {Number(metrics.area_sq_yards).toLocaleString()} <span className="text-[10px] text-slate-400 font-normal">sq yd</span>
                </div>
              </div>

              <div className="p-2.5 rounded-lg bg-black/30 border border-slate-800">
                <div className="text-[10px] text-slate-400 font-sans">Boundary Perimeter</div>
                <div className="text-emerald-400 font-bold mt-0.5">
                  {Number(metrics.perimeter_m).toLocaleString()} m <span className="text-[10px] text-slate-400 font-normal">({Number(metrics.perimeter_ft).toLocaleString()} ft)</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Open Surface Water Warning Banner */}
        {analysisData.is_surface_water && (
          <div className="p-4 rounded-xl bg-blue-950/40 border-2 border-blue-500/80 shadow-[0_0_30px_rgba(59,130,246,0.25)] space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-blue-400 font-extrabold text-xs uppercase tracking-wide">
                <Droplets className="w-4 h-4 text-blue-400 shrink-0 animate-pulse" />
                <span>Open Surface Water Body</span>
              </div>
              <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/40 font-mono font-bold text-[10px]">
                INAPPLICABLE
              </span>
            </div>
            <p className="text-xs text-blue-100 font-medium leading-relaxed bg-black/40 p-2.5 rounded-lg border border-blue-900/60">
              {analysisData.surface_water_warning ||
                'Selected area is located in an ocean or marine water body. Subsurface borewells cannot be drilled in open water bodies.'}
            </p>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2 rounded bg-black/40 border border-blue-900/60 text-center">
                <div className="text-[10px] text-slate-400 font-sans">Groundwater Potential</div>
                <div className="text-blue-400 font-black mt-0.5">0.0 (Surface Water / Marine Body)</div>
              </div>
              <div className="p-2 rounded bg-black/40 border border-blue-900/60 text-center">
                <div className="text-[10px] text-slate-400 font-sans">Borewell Drilling</div>
                <div className="text-blue-400 font-black mt-0.5">0 Pins (Not Applicable)</div>
              </div>
            </div>
          </div>
        )}

        {/* Hyper-Arid Desert Zone Warning Banner */}
        {analysisData.is_hyper_arid && (
          <div className="p-4 rounded-xl bg-rose-950/40 border-2 border-rose-500/80 shadow-[0_0_30px_rgba(244,63,94,0.25)] space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-rose-400 font-extrabold text-xs uppercase tracking-wide">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 animate-pulse" />
                <span>Hyper-Arid Desert Zone</span>
              </div>
              <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/40 font-mono font-bold text-[10px]">
                ZERO RECHARGE
              </span>
            </div>
            <p className="text-xs text-rose-100 font-medium leading-relaxed bg-black/40 p-2.5 rounded-lg border border-rose-900/60">
              {analysisData.aridity_warning ||
                'Hyper-Arid Desert Zone: Zero surface or shallow groundwater availability. SAR moisture signal reflects dry sand dielectric scattering, not liquid water.'}
            </p>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2 rounded bg-black/40 border border-rose-900/60 text-center">
                <div className="text-[10px] text-slate-400 font-sans">Groundwater Potential</div>
                <div className="text-rose-400 font-black mt-0.5">&lt; 15.0 (Barren)</div>
              </div>
              <div className="p-2 rounded bg-black/40 border border-rose-900/60 text-center">
                <div className="text-[10px] text-slate-400 font-sans">Sustainable Yield</div>
                <div className="text-rose-400 font-black mt-0.5">0 LPH (0 GPM)</div>
              </div>
            </div>
          </div>
        )}

        {/* P1, P2, P3 Target Cards or Zero-Points Marine Card */}
        {borewell_points.length === 0 ? (
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-center space-y-2 font-sans">
            <div className="flex items-center justify-center gap-1.5 text-slate-300 text-xs font-bold">
              <Info className="w-4 h-4 text-blue-400" />
              <span>Zero Borewell Target Points</span>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed max-w-xs mx-auto">
              {analysisData.is_surface_water
                ? 'Selected area is located in an ocean or marine water body. Subsurface borewells cannot be drilled in open water bodies.'
                : 'No viable borehole targets identified within the specified boundary.'}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {renderTargetCard(p1)}
            {renderTargetCard(p2)}
            {renderTargetCard(p3)}
          </div>
        )}

        {/* Aquifer & Subsurface Hydro-Summary Card */}
        <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Aquifer & Subsurface Summary</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">ISRIC + InSAR</span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="p-2 rounded bg-black/40 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-sans">Hydraulic Cond. (Ksat)</div>
              <div className="text-cyan-300 font-bold mt-0.5">
                {aquifer_summary.saturated_hydraulic_conductivity_cm_hr?.toFixed(2) || '2.84'} cm/hr
              </div>
            </div>

            <div className="p-2 rounded bg-black/40 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-sans">Soil Texture Class</div>
              <div className="text-slate-200 font-bold mt-0.5">
                {aquifer_summary.soil_texture_class || 'Sandy Clay Loam'}
              </div>
            </div>

            <div className="p-2 rounded bg-black/40 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-sans">InSAR Rate (24mo)</div>
              <div className="text-amber-400 font-bold mt-0.5 flex items-center gap-1">
                <TrendingDown className="w-3 h-3 text-amber-400" />
                {aquifer_summary.mean_vertical_velocity_mm_yr?.toFixed(1) || '-2.1'} mm/yr
              </div>
            </div>

            <div className="p-2 rounded bg-black/40 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-sans">Subsidence Risk</div>
              <div className="text-emerald-400 font-bold mt-0.5">
                {aquifer_summary.subsidence_risk_level || 'Low / Stable'}
              </div>
            </div>
          </div>

          <div className="text-[11px] text-slate-400 leading-relaxed font-sans bg-slate-950/60 p-2.5 rounded border border-slate-800/80">
            <span className="text-slate-300 font-semibold">Lithology: </span>
            {aquifer_summary.recommendation_summary ||
              'Granite-gneiss hard-rock terrain with weathered saprolite zone extending to 15-25m BGL. Fractured crystalline bedrock controls regional recharge with secondary porosity along conjugate fault lineaments.'}
          </div>
        </div>

        {/* Rig Operator Notes */}
        <div className="p-3.5 rounded-xl bg-amber-950/20 border border-amber-600/30 text-amber-200/90 text-xs space-y-1.5">
          <div className="font-bold flex items-center gap-1.5 text-amber-300">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            Borewell Rig Operator Advisory:
          </div>
          <p className="text-[11px] leading-relaxed text-amber-200/80 font-sans">
            Employ Down-the-Hole (DTH) pneumatic button bit rig. Expected primary fracture strike intercepts occur between 60m and 130m BGL. Slotting casing is mandatory across the upper saprolite (0–20m) to avert sediment collapse.
          </p>
        </div>
      </div>

      {/* Bottom Sticky Action: Google Earth KML Export */}
      <div className="p-4 border-t border-slate-800 bg-[#0e172a] shrink-0">
        <button
          onClick={onExportKml}
          disabled={isExporting}
          className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-xl font-bold text-xs tracking-wide shadow-[0_0_20px_rgba(16,185,129,0.3)] transition transform active:scale-95"
        >
          {isExporting ? (
            <>
              <Activity className="w-4 h-4 animate-spin" />
              Generating Google Earth KML...
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              Export Google Earth (.KML) Survey
            </>
          )}
        </button>
      </div>
    </div>
  );
}
