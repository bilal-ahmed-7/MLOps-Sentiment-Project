"""Schema and data-quality checks for the raw sentiment dataset."""
import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:  # older pandera versions
    import pandera as pa

from pandera.errors import SchemaErrors


class DataValidationError(Exception):
    """Raised when the dataset fails a mandatory validation check."""


def build_schema(cfg: dict) -> "pa.DataFrameSchema":
    """Build the Pandera schema from the pipeline config."""
    data = cfg["data"]
    min_len = cfg["validation"]["min_text_length"]
    return pa.DataFrameSchema(
        {
            data["text_column"]: pa.Column(
                str,
                nullable=False,
                checks=pa.Check(
                    lambda s: s.str.strip().str.len() >= min_len,
                    error=f"text shorter than {min_len} characters",
                ),
            ),
            data["label_column"]: pa.Column(
                str,
                nullable=False,
                checks=pa.Check.isin(list(data["label_map"])),
            ),
        },
        strict=False,  # extra columns are allowed
    )


def validate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Validate the dataframe. Returns it unchanged, or raises DataValidationError."""
    min_rows = cfg["validation"]["expected_min_rows"]
    if len(df) < min_rows:
        raise DataValidationError(
            f"Dataset has {len(df)} rows, expected at least {min_rows}."
        )
    try:
        build_schema(cfg).validate(df, lazy=True)
    except SchemaErrors as exc:
        summary = exc.failure_cases.head(10).to_string()
        raise DataValidationError(
            f"Schema validation failed ({len(exc.failure_cases)} issues):\n{summary}"
        ) from exc
    return df


def quality_report(df: pd.DataFrame, cfg: dict) -> dict:
    """Summary statistics, useful for logs and later for drift baselines."""
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]
    return {
        "n_rows": int(len(df)),
        "n_duplicate_texts": int(df.duplicated(subset=[text_col]).sum()),
        "label_distribution": df[label_col].value_counts(normalize=True).round(3).to_dict(),
        "avg_text_length": round(float(df[text_col].str.len().mean()), 1),
    }