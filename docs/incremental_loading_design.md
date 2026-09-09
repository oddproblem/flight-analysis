# Incremental Loading Notes

The current repo still loads the January 2024 pilot file. The loader can take another Parquet path, but automatic month-by-month ingestion is not implemented yet.

If I extend the project, I would add a small load-control table like this:

```sql
CREATE TABLE warehouse.load_batch (
    batch_id BIGSERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    source_sha256 CHAR(64) NOT NULL UNIQUE,
    reporting_year SMALLINT NOT NULL,
    reporting_month SMALLINT NOT NULL,
    source_row_count BIGINT NOT NULL,
    loaded_row_count BIGINT NOT NULL,
    load_started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    load_completed_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    error_message TEXT
);
```

I would use a SHA-256 checksum instead of only the file name. That way the same source file cannot be loaded twice just because it was renamed.

A monthly run would roughly be:

1. find a source file that has not been completed before;
2. validate its columns and month;
3. clean it and write Parquet;
4. update the date, airline, and airport dimensions;
5. insert new flight facts using the scheduled-flight natural key;
6. compare source and warehouse counts before commit;
7. mark the batch as complete only after the checks pass.

If a load fails, I would keep the failed batch record and error message so I can see what happened and retry it after fixing the problem.

This file is only a design note for now. The batch table and automatic monthly discovery are not part of the current implementation.
