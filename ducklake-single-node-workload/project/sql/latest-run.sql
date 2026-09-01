SELECT *
FROM ducklake_workload_lab.benchmark.measurements
WHERE run_id = (
    SELECT run_id
    FROM ducklake_workload_lab.benchmark.measurements
    ORDER BY recorded_at DESC
    LIMIT 1
)
ORDER BY step_order;
