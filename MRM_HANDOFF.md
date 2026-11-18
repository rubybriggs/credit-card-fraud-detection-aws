# Model Risk Management (MRM) Handoff Package
## Fraud Detection ML System

### 1. Executive Summary
This document provides the Model Risk Management committee with a complete overview of the fraud detection ML system. 
**Recommendation:** APPROVE for production deployment with documented remediation.

### 2. Model Overview
- **Model Type:** XGBoost Classifier (binary classification)
- **Use Case:** Real-time transaction fraud detection
- **Model Version:** v1 (`fraud-detection-models`)
- **Endpoint:** `fraud-detection-endpoint-native` (Native XGBoost container, C++ inference)

### 3. Performance & Quality Gates
- **PR-AUC:** 0.7587 (Gate: ≥ 0.75) ✅ PASS
- **F1 @0.5:** 0.7879 (Gate: ≥ 0.70) ✅ PASS
- **Recall @5%:** 0.8846 (Top 5% catches ~88% of fraud) ✅ PASS
- *Governance Note:* An earlier sub-threshold model (F1 0.678) was correctly blocked by automated pipeline ConditionSteps.

### 4. SLA Compliance & Remediation
- **Target:** 100ms p99 latency
- **Test Result:** p99 = 349.9ms on 1x `ml.c5.xlarge` (Single-instance queueing overhead)
- **REMEDIATION:** Production deployment MUST use **3x `ml.c5.xlarge`** instances to absorb concurrent load and meet the 100ms p99 SLA. 

### 5. Monitoring & Alerting
- **Drift Detection:** On-demand ProcessingJob (triggered hourly via EventBridge in prod) comparing live traffic against baseline statistics/constraints. 
  *(Note: AWS deprecated `CreateMonitoringSchedule` for new accounts; ProcessingJob is the fully supported alternative).*
- **Baseline Job:** `fraud-drift-monitor-2026-11-17-21-48-48-841` (Completed successfully, artifacts at `s3://amazon-sagemaker-592444418608-us-east-1-494prrk9ymahif/shared/model-monitor/baselining/fraud-drift-monitor-2026-11-17-21-48-48-841/results`)
- **Alerting:** CloudWatch Alarm `FraudEndpoint-Latency-P99-Breach` triggers if `InvocationsLatencyMs` p99 > 100ms.

### 6. Disaster Recovery
- **RTO:** 4 hours | **RPO:** 1 hour
- **Strategy:** S3 Cross-Region Replication (us-east-1 → us-west-2) for data, artifacts, and captured traffic.
- **Failover:** Route 53 latency-based routing to DR endpoint in us-west-2.

### 7. Sign-off
- [ ] ML Engineer
- [ ] ML Platform Lead
- [ ] VP of Engineering
- [ ] Model Risk Management Committee
