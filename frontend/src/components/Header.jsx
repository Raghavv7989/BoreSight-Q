import React from 'react';
import { Shield, Cpu, Satellite, Radio, Info } from 'lucide-react';

export default function Header({ bufferM, onBufferChange }) {
  return (
    <header className="h-14 bg-[#0a0f18] border-b border-slate-800 px-5 flex items-center justify-between shrink-0 select-none z-30">
      {/* Brand & Title */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-600 via-blue-600 to-emerald-500 p-0.5 shadow-[0_0_15px_rgba(0,229,255,0.3)]">
          <div className="w-full h-full bg-[#0b101b] rounded-[10px] flex items-center justify-center">
            <Radio className="w-5 h-5 text-cyan-400 animate-pulse" />
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-black tracking-wider text-white font-sans">
              BORESIGHT<span className="text-cyan-400 font-mono">-Q</span>
            </h1>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 font-bold">
              v1.0 HYBRID-QPU
            </span>
          </div>
          <p className="text-[10px] text-slate-400 hidden sm:block">
            Parcel-Scale Satellite Hydrogeology & Quantum Variational Borewell Siting
          </p>
        </div>
      </div>

      {/* Center Engine Telemetry Badges */}
      <div className="hidden md:flex items-center gap-3 text-[11px] font-mono">
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
          <Satellite className="w-3.5 h-3.5 text-cyan-400" />
          <span>Copernicus + S1/S2 STAC</span>
        </div>
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
          <span>12-Qubit VQR + AHP</span>
        </div>
      </div>

      {/* Right Controls: Buffer selector */}
      <div className="flex items-center gap-3 text-xs">
        <div className="flex items-center gap-1.5 bg-slate-900 px-2.5 py-1 rounded-lg border border-slate-800 text-slate-300 font-mono">
          <span className="text-slate-400 text-[11px]">Buffer:</span>
          <select
            value={bufferM}
            onChange={(e) => onBufferChange(parseFloat(e.target.value))}
            className="bg-transparent text-white focus:outline-none cursor-pointer text-xs"
          >
            <option value="500" className="bg-slate-900 text-white">500 m</option>
            <option value="1000" className="bg-slate-900 text-white">1,000 m</option>
            <option value="1500" className="bg-slate-900 text-white">1,500 m</option>
            <option value="2000" className="bg-slate-900 text-white">2,000 m</option>
          </select>
        </div>
      </div>
    </header>
  );
}
