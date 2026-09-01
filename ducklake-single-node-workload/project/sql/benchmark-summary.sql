SELECT
    profile,
    phase,
    workload,
    count(DISTINCT run_id) AS runs,
    round(median(elapsed_ms), 3) AS median_ms,
    round(min(elapsed_ms), 3) AS min_ms,
    round(max(elapsed_ms), 3) AS max_ms,
    round(median(rows_per_second), 2) AS median_rows_per_second,
    max(peak_rss_bytes) AS max_peak_rss_bytes,
    max(output_bytes) AS max_managed_bytes,
    bool_and(correctness) AS all_correct
FROM ducklake_workload_lab.benchmark.measurements
WHERE phase = 'measured'
GROUP BY profile, phase, workload
ORDER BY
    CASE profile WHEN 'quick' THEN 1 WHEN 'small' THEN 2 WHEN 'medium' THEN 3 WHEN 'large' THEN 4 ELSE 5 END,
    min(step_order);
