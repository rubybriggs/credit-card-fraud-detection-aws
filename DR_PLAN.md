# Disaster Recovery — Fraud Detection ML System
**RTO:** 4 hours | **RPO:** 1 hour

## 1. Data replication
- S3 Cross-Region Replication (CRR) us-east-1 -> us-west-2 for:
  - `fraud-detection-processed-592444418608-prod` (masked data, preprocessed splits, datacapture)
  - `fraud-detection-artifacts-592444418608-prod` (model artifacts, inference code)
- KMS: use multi-region keys for seamless cross-region decrypt
- Model Registry is region-scoped: re-register latest approved package in us-west-2 from replicated artifacts

## 2. Endpoint failover
- Primary: us-east-1 `fraud-detection-endpoint-native` (3x ml.c5.xlarge)
- DR: us-west-2 (3x ml.c5.xlarge, provisioned via CloudFormation)
- Route 53 latency-based routing + health checks; auto-shift on 5xx/timeout
- Failover time: ~15 min (endpoint create + warmup)

## 3. Pipeline recovery (DR event)
1. Deploy CloudFormation stack in us-west-2
2. Re-run pipeline from last successful execution (data replicated via CRR)
3. Approve latest model version in us-west-2 registry
4. Deploy endpoint in us-west-2

## 4. Testing cadence
- Quarterly DR drill (simulate us-east-1 outage)
- Annual 1-hour full production cutover to us-west-2

## 5. Escalation
On-call ML Engineer -> ML Platform Lead -> VP of Engineering
