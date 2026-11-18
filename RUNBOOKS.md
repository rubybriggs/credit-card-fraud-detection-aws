# Operational Runbooks — Fraud Detection ML System

## Runbook 1: Endpoint latency SLA breach
**Trigger:** CloudWatch `InvocationsLatencyMs` p99 > 100ms for 5 min
1. Confirm via CloudWatch (server-side metric, not client-side)
2. Scale out: `update_endpoint_weights_and_capacities(EndpointName='fraud-detection-endpoint-native', DesiredWeightsAndCapacities=[{'DesiredInstanceCount': 6}])`
3. If still high: switch to ml.c5.2xlarge
4. Re-run load test; document root cause

## Runbook 2: Model drift detected
**Trigger:** drift report `violations` > 0
1. Review `s3://fraud-detection-processed-592444418608-fraud/monitor/drift-reports/`
2. Severity: <10% features = monitor; 10-30% = retrain pipeline; >30% = rollback to prior approved version
3. Retrain: re-run pipeline; approve only if gates pass
4. Deploy new model version

## Runbook 3: Endpoint 5xx errors
**Trigger:** `Invocations` 5xx > 10 for 5 min
1. `describe_endpoint` -> if Failed, delete + recreate from CloudFormation
2. If persistent, failover to us-west-2 DR endpoint
3. Post-incident: document

## Runbook 4: Pipeline execution failure
**Trigger:** pipeline execution status = Failed
1. `list_pipeline_execution_steps` -> find failed step
2. Read CloudWatch logs of that Processing/Training job
3. Fix (data/code/resource); re-run pipeline

## On-call
Primary: ML Engineer (weekly rotation) | Secondary: Platform Lead | Escalation: VP Eng
