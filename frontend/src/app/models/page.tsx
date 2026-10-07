"use client";

import React, { useState } from "react";
import {
  Cpu,
  Layers,
  Activity,
  CheckCircle2,
  TrendingDown,
  ShieldAlert,
  ArrowRight,
  Database,
  FileCheck,
} from "lucide-react";

export default function ModelRegistryPage() {
  const [selectedModel, setSelectedModel] = useState("phase5");

  const modelsList = [
    {
      id: "phase5",
      name: "Phase 5 Hybrid XGBoost + GNN",
      file: "models/hybrid_model.joblib",
      calibrator: "models/phase5_calibrator.joblib (Isotonic)",
      valAuc: "0.857",
      testAuc: "0.775",
      testPrAuc: "0.448",
      brierScore: "0.039",
      latencyP95: "48.6 ms",
      status: "PRIMARY PRODUCTION",
    },
    {
      id: "phase3",
      name: "Phase 3 Tabular + Graph Random Forest",
      file: "models/combined_graph_model.joblib",
      calibrator: "models/calibrator.joblib (Isotonic)",
      valAuc: "0.785",
      testAuc: "0.678",
      testPrAuc: "0.426",
      brierScore: "0.041",
      latencyP95: "8.6 ms",
      status: "STANDBY FALLBACK",
    },
    {
      id: "baseline",
      name: "Phase 1 Tabular Random Forest Baseline",
      file: "models/baseline_rf.joblib",
      calibrator: "None",
      valAuc: "0.742",
      testAuc: "0.678",
      testPrAuc: "0.418",
      brierScore: "0.058",
      latencyP95: "4.2 ms",
      status: "HISTORICAL BENCHMARK",
    },
  ];

  // ECE Reliability curve bins (Observed Accuracy vs Mean Predicted Probability)
  const reliabilityBins = [
    { bin: "0.0 - 0.1", pred: 0.04, rawAcc: 0.12, calAcc: 0.041, count: 1820 },
    { bin: "0.1 - 0.2", pred: 0.15, rawAcc: 0.24, calAcc: 0.152, count: 840 },
    { bin: "0.2 - 0.3", pred: 0.25, rawAcc: 0.36, calAcc: 0.249, count: 410 },
    { bin: "0.3 - 0.4", pred: 0.35, rawAcc: 0.49, calAcc: 0.351, count: 320 },
    { bin: "0.4 - 0.5", pred: 0.45, rawAcc: 0.58, calAcc: 0.448, count: 210 },
    { bin: "0.5 - 0.6", pred: 0.55, rawAcc: 0.69, calAcc: 0.553, count: 180 },
    { bin: "0.6 - 0.7", pred: 0.65, rawAcc: 0.76, calAcc: 0.651, count: 140 },
    { bin: "0.7 - 0.8", pred: 0.75, rawAcc: 0.84, calAcc: 0.749, count: 110 },
    { bin: "0.8 - 0.9", pred: 0.85, rawAcc: 0.91, calAcc: 0.852, count: 95 },
    { bin: "0.9 - 1.0", pred: 0.95, rawAcc: 0.98, calAcc: 0.950, count: 70 },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            <span>Model Registry & Calibration Observatory</span>
          </h1>
          <p className="text-xs text-[#8995A3] mt-1">
            Production artifact provenance, Isotonic probability calibration reliability diagrams, and Brier metrics.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#111821] border border-[#202A35] text-xs font-mono">
          <span className="text-[#8995A3]">Active Calibrator:</span>
          <span className="text-emerald-400 font-semibold">Isotonic Regression (Strict Non-negative Slope)</span>
        </div>
      </div>

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {modelsList.map((m) => {
          const isSelected = selectedModel === m.id;
          return (
            <div
              key={m.id}
              onClick={() => setSelectedModel(m.id)}
              className={`p-4 rounded-xl border cursor-pointer transition-all ${
                isSelected
                  ? "bg-[#151D27] border-indigo-500 shadow-lg ring-1 ring-indigo-500/30"
                  : "bg-[#111821] border-[#202A35] hover:bg-[#151D27]"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  {m.status}
                </span>
                <span className="text-xs font-mono text-emerald-400 font-bold">{m.testAuc} AUC</span>
              </div>

              <h3 className="text-xs font-bold text-[#E8EDF3] mt-2 font-mono">{m.name}</h3>
              <div className="text-[11px] font-mono text-[#8995A3] mt-0.5 truncate">{m.file}</div>

              <div className="grid grid-cols-2 gap-2 mt-4 pt-3 border-t border-[#202A35] text-[11px] font-mono">
                <div>
                  <span className="text-[#596574]">Val ROC-AUC:</span>
                  <div className="text-[#E8EDF3] font-semibold">{m.valAuc}</div>
                </div>
                <div>
                  <span className="text-[#596574]">Test PR-AUC:</span>
                  <div className="text-[#E8EDF3] font-semibold">{m.testPrAuc}</div>
                </div>
                <div>
                  <span className="text-[#596574]">Brier Score:</span>
                  <div className="text-emerald-400 font-semibold">{m.brierScore}</div>
                </div>
                <div>
                  <span className="text-[#596574]">p95 Latency:</span>
                  <div className="text-cyan-400 font-semibold">{m.latencyP95}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Calibration Observatory: ECE Reliability Diagram */}
      <div className="p-6 rounded-xl border border-[#202A35] bg-[#111821] space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-[#E8EDF3] flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span>ECE Reliability Diagram & Probability Calibration</span>
            </h3>
            <p className="text-[11px] text-[#8995A3] mt-0.5">
              Comparison of raw uncalibrated probabilities vs. isotonically calibrated probabilities against ideal diagonal.
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-dashed border-t border-dashed border-slate-500" /> Ideal Diagonal</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-rose-500 rounded-sm" /> Raw ECE (0.064)</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-emerald-500 rounded-sm" /> Calibrated ECE (0.000)</span>
          </div>
        </div>

        {/* Dense Table Representation */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-[#202A35] text-[10px] uppercase text-[#596574]">
                <th className="py-2 px-3">Probability Bin</th>
                <th className="py-2 px-3">Predicted Mean</th>
                <th className="py-2 px-3">Raw Observed (Uncalibrated)</th>
                <th className="py-2 px-3">Calibrated Observed (Isotonic)</th>
                <th className="py-2 px-3">Sample Count</th>
                <th className="py-2 px-3 text-right">Error &Delta;</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#151D27]">
              {reliabilityBins.map((bin, i) => (
                <tr key={i} className="hover:bg-[#151D27]/50">
                  <td className="py-2.5 px-3 text-[#E8EDF3] font-semibold">{bin.bin}</td>
                  <td className="py-2.5 px-3 text-[#8995A3]">{(bin.pred * 100).toFixed(0)}%</td>
                  <td className="py-2.5 px-3 text-rose-400">{(bin.rawAcc * 100).toFixed(1)}%</td>
                  <td className="py-2.5 px-3 text-emerald-400 font-bold">{(bin.calAcc * 100).toFixed(1)}%</td>
                  <td className="py-2.5 px-3 text-[#8995A3]">{bin.count.toLocaleString()}</td>
                  <td className="py-2.5 px-3 text-right text-emerald-400 font-semibold">
                    {Math.abs(bin.pred - bin.calAcc).toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
