-- Dynamic edge: the today-relative fields, computed at query time over the
-- static core table `dim_date`. A view needs no refresh job -- its expressions
-- re-evaluate on every query, so "today" is always correct without rewriting a
-- single stored row. (Syntax shown for a lakehouse SQL dialect.)

CREATE VIEW v_dim_date_dynamic AS
SELECT
    d.*,
    d.date_value = current_date()                       AS is_today,
    d.date_value <  current_date()                      AS is_past,
    date_trunc('month', d.date_value)
        = date_trunc('month', current_date())           AS is_current_month,
    datediff(d.date_value, current_date())              AS days_from_today
FROM dim_date d;

-- Reports that need "current month" query this view. Reports that need a fixed
-- historical period join the static core on the date KEY and are immune to what
-- day it is.

-- Timezone note: current_date() is UTC in most lakehouse SQL contexts. If your
-- business day rolls over at local midnight, anchor the boundary explicitly:
--   to_date(from_utc_timestamp(current_timestamp(), 'America/Toronto'))

-- Serving-mode note: some engines (e.g. Direct Lake) won't query a SQL view at
-- all -- there, push these same relative fields into measures that call TODAY().
