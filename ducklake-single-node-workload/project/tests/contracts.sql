-- name: Every retained measurement passed its correctness gate
SELECT run_id, workload
FROM ducklake_workload_lab.benchmark.measurements
WHERE NOT correctness;

-- name: Every complete Run retained seven measured workload steps
SELECT run_id, count(*) AS actual_steps
FROM ducklake_workload_lab.benchmark.measurements
GROUP BY run_id
HAVING count(*) <> 7;

-- name: Current order fact has a unique business key
SELECT order_id
FROM ducklake_workload_lab.analytics.fact_orders
GROUP BY order_id
HAVING count(*) > 1;

-- name: Current event population reconciles with quarantine
SELECT 1
WHERE (SELECT count(*) FROM ducklake_workload_lab.raw.events)
   <> (SELECT count(*) FROM ducklake_workload_lab.staging.events)
     + (SELECT count(*) FROM ducklake_workload_lab.staging.events_quarantine);
