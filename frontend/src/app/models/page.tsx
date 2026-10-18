"use client";

import React, { useState, useEffect } from "react";
import { Cpu } from "lucide-react";

interface MetricSummary {
  roc_mean: number;
  pr_mean: number;
}

interface PairedLift {
  mean_diff: number;
  p_value: number;
}

interface PairedRocDiff {
  roc: { mean_diff: number };
}

interface AblationMetric {
  roc_auc: number;
  pr_auc: number;
}

interface ReturnFraudAblations {
  full_model: AblationMetric;
  without_days_to_return: AblationMetric;
  without_reasons: AblationMetric;
  without_both: AblationMetric;
}

interface PriceAnomalySubgroups {
  test_fake_with_price_anomaly: number;
  recall_with_price_anomaly: number;
  test_fake_without_price_anomaly: number;
  recall_without_price_anomaly: number;
}

interface DesignRuleAblations {
  return_fraud_ablations: ReturnFraudAblations;
  fake_listing_price_anomaly_subgroups: PriceAnomalySubgroups;
}

interface RegistryResultsData {
  pre_registered_primary_analysis_20_seeds?: {
    tabular_plus_graph: MetricSummary;
    tabular_with_device: MetricSummary;
    primary_roc_lift_paired: PairedLift;
  };
  graph_free_baseline_comparisons_20_seeds?: {
    device_lift_b_vs_a_paired: PairedRocDiff;
  };
  tuned_hybrid_model_5_seeds?: {
    standard_dataset?: {
      roc_mean: number;
      pr_mean: number;
      paired_diff_vs_tabular: PairedRocDiff;
    };
  };
  design_rule_ablations?: DesignRuleAblations;
  [key: string]: unknown;
}

export default function ModelRegistryPage() {
  const [selectedModel, setSelectedModel] = useState("tabular_graph");
  const [resultsData, setResultsData] = useState<RegistryResultsData | null>(null);

  useEffect(() => {
    fetch("/results.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load results.json");
        return res.json();
      })
      .then((data) => setResultsData(data))
      .catch((err) => console.error("Could not fetch results.json:", err));
  }, []);

  /* BEGIN GENERATED */
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
      badgeColor: "text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
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
      badgeColor: "text-blue-600 dark:text-blue-400 bg-blue-500/10 border-blue-500/20",
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
      badgeColor: "text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20",
    },
  ];
  /* END GENERATED */

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--text-primary)] flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            <span>Model Registry &amp; Audited Performance</span>
          </h1>
          <p className="text-xs text-[var(--text-secondary)] mt-1">
            Dynamic registry rendering audited metrics loaded from <code className="text-indigo-600 dark:text-indigo-400 font-mono">results/results.json</code>.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--surface)] border border-[var(--border)] text-xs font-mono">
          <span className="text-[var(--text-muted)]">Registry Sync:</span>
          <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Live from results/results.json</span>
        </div>
      </div>

      {/* Model Cards Grid */}
      {/* BEGIN GENERATED */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {modelsList.map((m) => {
          const isSelected = selectedModel === m.id;
          return (
            <div
              key={m.id}
              onClick={() => setSelectedModel(m.id)}
              className={`p-4 rounded-xl border cursor-pointer transition-all ${
                isSelected
                  ? "bg-[var(--surface-elevated)] border-blue-500 shadow-md ring-1 ring-blue-500/30"
                  : "bg-[var(--surface)] border-[var(--border)] hover:bg-[var(--surface-elevated)]"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded border font-semibold ${m.badgeColor}`}>
                  {m.status}
                </span>
                <span className="text-xs font-mono text-[var(--text-muted)]">{m.calibrator}</span>
              </div>

              <div className="mt-3">
                <h3 className="text-sm font-bold text-[var(--text-primary)]">{m.name}</h3>
                <div className="text-[11px] font-mono text-[var(--text-muted)] mt-0.5">{m.file}</div>
              </div>

              <div className="grid grid-cols-2 gap-2 mt-4 pt-3 border-t border-[var(--border)] text-xs font-mono">
                <div>
                  <span className="text-[10px] text-[var(--text-muted)] block">Test ROC-AUC</span>
                  <span className="text-base font-bold text-blue-600 dark:text-blue-400">{m.testAuc}</span>
                </div>
                <div>
                  <span className="text-[10px] text-[var(--text-muted)] block">Test PR-AUC</span>
                  <span className="text-base font-bold text-purple-600 dark:text-purple-400">{m.testPrAuc}</span>
                </div>
              </div>

              <div className="mt-3 pt-2 border-t border-[var(--border)] flex items-center justify-between text-[11px] font-mono">
                <span className="text-[var(--text-muted)]">Paired Difference</span>
                <span className="text-[var(--text-primary)] font-semibold">{m.lift}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Design-Rule Ablation Summary Cards */}
      {ablations && ablations.return_fraud_ablations && ablations.fake_listing_price_anomaly_subgroups && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-[var(--text-primary)] uppercase tracking-wider text-[11px]">
                Return Fraud Detector Ablations (Uncensored Test N=1,100)
              </h3>
              <span className="text-[10px] text-amber-600 dark:text-amber-400 font-semibold">Rule Recovery</span>
            </div>
            <div className="space-y-1.5 text-[var(--text-secondary)]">
              <div className="flex justify-between p-2 rounded bg-[var(--surface-elevated)] border border-[var(--border)]">
                <span>Full Model (Days + Reasons + Tabular)</span>
                <span className="text-emerald-600 dark:text-emerald-400 font-bold">ROC: {ablations.return_fraud_ablations.full_model.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.full_model.pr_auc.toFixed(3)}</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-[var(--surface-elevated)] border border-[var(--border)]">
                <span>Without days_to_return</span>
                <span className="text-amber-600 dark:text-amber-400 font-bold">ROC: {ablations.return_fraud_ablations.without_days_to_return.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.without_days_to_return.pr_auc.toFixed(3)}</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-[var(--surface-elevated)] border border-[var(--border)]">
                <span>Without reason_* features</span>
                <span className="text-blue-600 dark:text-blue-400 font-bold">ROC: {ablations.return_fraud_ablations.without_reasons.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.without_reasons.pr_auc.toFixed(3)}</span>
              </div>
              <div className="flex justify-between p-2 rounded bg-[var(--surface-elevated)] border border-[var(--border)]">
                <span>Without Both (Delay &amp; Reasons)</span>
                <span className="text-rose-600 dark:text-rose-400 font-bold">ROC: {ablations.return_fraud_ablations.without_both.roc_auc.toFixed(3)} · PR: {ablations.return_fraud_ablations.without_both.pr_auc.toFixed(3)}</span>
              </div>
            </div>
          </div>

          <div className="p-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-md space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-[var(--text-primary)] uppercase tracking-wider text-[11px]">
                Fake Listing Subgroup Recall (Price Anomaly Split)
              </h3>
              <span className="text-[10px] text-amber-600 dark:text-amber-400 font-semibold">Rule Recovery</span>
            </div>
            <div className="space-y-3 text-[var(--text-secondary)] pt-2">
              <div className="p-3 rounded bg-[var(--surface-elevated)] border border-[var(--border)] flex items-center justify-between">
                <div>
                  <div className="text-[var(--text-primary)] font-semibold">With Price Anomaly (N={ablations.fake_listing_price_anomaly_subgroups.test_fake_with_price_anomaly})</div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-0.5">Price &lt; 0.6 catalog base price</div>
                </div>
                <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400">
                  {(ablations.fake_listing_price_anomaly_subgroups.recall_with_price_anomaly * 100).toFixed(1)}% Recall
                </div>
              </div>
              <div className="p-3 rounded bg-[var(--surface-elevated)] border border-[var(--border)] flex items-center justify-between">
                <div>
                  <div className="text-[var(--text-primary)] font-semibold">Without Price Anomaly (N={ablations.fake_listing_price_anomaly_subgroups.test_fake_without_price_anomaly})</div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-0.5">Non-anomalous price listings</div>
                </div>
                <div className="text-lg font-bold text-rose-600 dark:text-rose-400">
                  {(ablations.fake_listing_price_anomaly_subgroups.recall_without_price_anomaly * 100).toFixed(1)}% Recall
                </div>
              </div>
              <div className="text-[11px] text-[var(--text-muted)] italic">
                * Note: Evaluated on 20-seed holdout benchmark. Confirms tabular listing detector identifies explicit pricing distortion rules.
              </div>
            </div>
          </div>
        </div>
      )}
      {/* END GENERATED */}
    </div>
  );
}
