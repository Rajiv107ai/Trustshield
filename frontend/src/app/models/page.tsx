"use client";

import React, { useState, useEffect } from "react";
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
  const [selectedModel, setSelectedModel] = useState("tabular_graph");
  const [resultsData, setResultsData] = useState<any>(null);

  useEffect(() => {
    fetch("/results.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load results.json");
        return res.json();
      })
      .then((data) => setResultsData(data))
      .catch((err) => console.error("Could not fetch results.json:", err));
  }, []);

  const primaryAnalysis = resultsData?.pre_registered_primary_analysis_20_seeds;
  const graphFree = resultsData?.graph_free_baseline_comparisons_20_seeds;
  const hybridModel = resultsData?.tuned_hybrid_model_5_seeds?.standard_dataset;
  const ablations = resultsData?.design_rule_ablations;

  const modelsList = [
    {
      id: "tabular_graph",
      name: "Tabular + Graph XGBoost (Primary)",
      file: "models/combined_graph_model.joblib",
      calibrator: "Isotonic Regression",
      testAuc: primaryAnalysis ? primaryAnalysis.tabular_plus_graph.roc_mean.toFixed(3) : "—",
      testPrAuc: primaryAnalysis ? primaryAnalysis.tabular_plus_graph.pr_mean.toFixed(3) : "—",
      lift: primaryAnalysis ? `+${primaryAnalysis.primary_roc_lift_paired.mean_diff.toFixed(3)} ROC (p=${primaryAnalysis.primary_roc_lift_paired.p_value.toFixed(4)})` : "—",
      status: "PRIMARY AUDITED CHAMPION",
      badgeColor: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    },
    {
      id: "tabular_with_dev",
      name: "Tabular Baseline (With Device Count)",
      file: "models/baseline_rf.joblib",
      calibrator: "Platt Scaling",
      testAuc: primaryAnalysis ? primaryAnalysis.tabular_with_device.roc_mean.toFixed(3) : "—",
      testPrAuc: primaryAnalysis ? primaryAnalysis.tabular_with_device.pr_mean.toFixed(3) : "—",
      lift: graphFree ? `+${graphFree.device_lift_b_vs_a_paired.roc.mean_diff.toFixed(3)} vs No-Device` : "Baseline",
      status: "CURRENT BASELINE (10 Feats)",
      badgeColor: "text-blue-400 bg-blue-500/10 border-blue-500/20",
    },
    {
      id: "hybrid",
      name: "Tuned Hybrid (OOF GraphSAGE + XGB)",
      file: "models/hybrid_model.joblib",
      calibrator: "Isotonic Regression",
      testAuc: hybridModel ? hybridModel.roc_mean.toFixed(3) : "—",
      testPrAuc: hybridModel ? hybridModel.pr_mean.toFixed(3) : "—",
      lift: hybridModel ? `${hybridModel.paired_diff_vs_tabular.roc.mean_diff.toFixed(3)} ROC vs Tabular` : "Underperforms",
      status: "EXPLORATORY HYBRID",
      badgeColor: "text-amber-400 bg-amber-500/10 border-amber-500/20",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#E8EDF3] flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            <span>Model Registry & Audited Performance</span>
          </h1>
          <p className="text-xs text-[#8995A3] mt-1">
            Dynamic registry rendering audited metrics loaded from <code className="text-indigo-400">results/results.json</code>.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#111821] border border-[#202A35] text-xs font-mono">
          <span className="text-[#8995A3]">Registry Sync:</span>
          <span className="text-emerald-400 font-semibold">Live from results/results.json</span>
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
                  ? "bg-[#151D27] border-blue-500/50 shadow-lg shadow-blue-500/5"
                  : "bg-[#111821] border-[#202A35] hover:border-[#2D3A4B]"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${m.badgeColor}`}>
                  {m.status}
                </span>
                <span className="text-xs font-mono text-[#596574]">{m.calibrator}</span>
              </div>

              <div className="mt-3">
                <h3 className="text-sm font-bold text-[#E8EDF3]">{m.name}</h3>
                <div className="text-[11px] font-mono text-[#596574] mt-0.5">{m.file}</div>
              </div>

              <div className="grid grid-cols-2 gap-2 mt-4 pt-3 border-t border-[#202A35]/60 text-xs font-mono">
                <div>
                  <span className="text-[10px] text-[#596574] block">Test ROC-AUC</span>
                  <span className="text-base font-bold text-blue-400">{m.testAuc}</span>
                </div>
                <div>
                  <span className="text-[10px] text-[#596574] block">Test PR-AUC</span>
                  <span className="text-base font-bold text-purple-400">{m.testPrAuc}</span>
                </div>
              </div>

              <div className="mt-3 pt-2 border-t border-[#202A35]/40 flex items-center justify-between text-[11px] font-mono">
                <span className="text-[#596574]">Paired Difference</span>
                <span className="text-[#E8EDF3] font-semibold">{m.lift}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Design-Rule Ablation Summary Cards */}
      {ablations && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-[#E8EDF3] uppercase tracking-wider text-[11px]">
                Return Fraud Detector Ablations (Uncensored Test N=1,100)
              </h3>
              <span className="text-[10px] text-amber-400">Generator Rule Recovery</span>
            </div>
            <div className="space-y-1.5 text-[#8995A3]">
              <div className="flex justify-between p-2 rounded bg-[#0E131A] border border-[#202A35]">
                <span>Full Model (Days + Reasons + Tabular)</span>
                <span className="text-emerald-400 font-bold">ROC: {ablations.return_fraud_ablations.full_model.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.full_model.pr_auc.toFixed(3)}</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-[#0E131A] border border-[#202A35]">
                <span>Without days_to_return</span>
                <span className="text-amber-400 font-bold">ROC: {ablations.return_fraud_ablations.without_days_to_return.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.without_days_to_return.pr_auc.toFixed(3)}</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-[#0E131A] border border-[#202A35]">
                <span>Without reason_* features</span>
                <span className="text-blue-400 font-bold">ROC: {ablations.return_fraud_ablations.without_reasons.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.without_reasons.pr_auc.toFixed(3)}</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-[#0E131A] border border-[#202A35]">
                <span>Without Both (Delay & Reasons)</span>
                <span className="text-rose-400 font-bold">ROC: {ablations.return_fraud_ablations.without_both.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.without_both.pr_auc.toFixed(3)}</span>
              </div>
            </div>
          </div>

          <div className="p-5 rounded-xl border border-[#202A35] bg-[#111821] space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-[#E8EDF3] uppercase tracking-wider text-[11px]">
                Fake Listing Subgroup Recall (Price Anomaly Split)
              </h3>
              <span className="text-[10px] text-amber-400">Generator Rule Recovery</span>
            </div>
            <div className="space-y-3 text-[#8995A3] pt-2">
              <div className="p-3 rounded bg-[#0E131A] border border-[#202A35] flex items-center justify-between">
                <div>
                  <div className="text-[#E8EDF3] font-semibold">With Injected Price Anomaly (N={ablations.fake_listing_price_anomaly_subgroups.test_fake_with_price_anomaly})</div>
                  <div className="text-[10px] text-[#596574] mt-0.5">Price &lt; 0.60 catalog base price</div>
                </div>
                <div className="text-lg font-bold text-emerald-400">
                  {(ablations.fake_listing_price_anomaly_subgroups.recall_with_price_anomaly * 100).toFixed(1)}% Recall
                </div>
              </div>
              <div className="p-3 rounded bg-[#0E131A] border border-[#202A35] flex items-center justify-between">
                <div>
                  <div className="text-[#E8EDF3] font-semibold">Without Injected Price Anomaly (N={ablations.fake_listing_price_anomaly_subgroups.test_fake_without_price_anomaly})</div>
                  <div className="text-[10px] text-[#596574] mt-0.5">Non-anomalous price listings</div>
                </div>
                <div className="text-lg font-bold text-rose-400">
                  {(ablations.fake_listing_price_anomaly_subgroups.recall_without_price_anomaly * 100).toFixed(1)}% Recall
                </div>
              </div>
              <div className="text-[11px] text-[#596574] italic">
                * Note: Demonstrates that the tabular listing detector primarily recovers the generator's explicit price anomaly rule rather than general counterfeit semantics.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
