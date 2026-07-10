-- A whole pipeline is one file. Steps run top to bottom; asserts gate the run.
-- (The runner loads the input into a table `raw` and emits the table `out`.)

--@pipeline load_orders

--@step clean
CREATE TABLE out AS SELECT id, amount FROM raw WHERE amount > 0

--@assert no_null_id EXPECT_NO_ROWS
SELECT * FROM out WHERE id IS NULL

--@assert has_rows EXPECT_TRUE
SELECT count(*) > 0 FROM out
