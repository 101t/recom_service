//! Tabular file handling (CSV, TSV, JSON, NDJSON, Parquet) and conversion of rows
//! into items and interactions.
//!
//! All functions here are blocking; call them from `spawn_blocking`.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use chrono::{DateTime, TimeZone, Utc};
use polars::prelude::*;
use serde_json::{Map, Value};

use crate::error::{AppError, AppResult};
use crate::ids::{MAX_ENTITY_ID_LEN, content_id};
use crate::models::{Interaction, Item};

/// Longest text (in characters) sent to the embedder for one item.
pub const MAX_TEXT_CHARS: usize = 20_000;

/// Column names tried, in order, when no id column is configured.
pub const ID_CANDIDATES: &[&str] = &[
    "id",
    "item_id",
    "itemid",
    "product_id",
    "productid",
    "sku",
    "_id",
    "uuid",
    "doc_id",
    "document_id",
];

/// Column names used (in this order) to build the embedded text when no text
/// columns are configured.
pub const TEXT_CANDIDATES: &[&str] = &[
    "title",
    "name",
    "headline",
    "text",
    "content",
    "description",
    "summary",
    "abstract",
    "body",
    "category",
    "categories",
    "topic",
    "genre",
    "tags",
    "keywords",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Csv,
    Tsv,
    /// A JSON array of objects, or NDJSON (detected from the content).
    Json,
    NdJson,
    Parquet,
}

impl FileFormat {
    pub fn from_filename(name: &str) -> Option<Self> {
        let ext = Path::new(name).extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "csv" => Some(Self::Csv),
            "tsv" | "tab" => Some(Self::Tsv),
            "json" => Some(Self::Json),
            "jsonl" | "ndjson" => Some(Self::NdJson),
            "parquet" | "pq" => Some(Self::Parquet),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Tsv => "tsv",
            Self::Json => "json",
            Self::NdJson => "jsonl",
            Self::Parquet => "parquet",
        }
    }
}

pub const SUPPORTED_FORMATS: &str = "CSV (.csv), TSV (.tsv), JSON (.json array or lines), NDJSON (.jsonl/.ndjson), Parquet (.parquet)";

/// Reads a tabular file into a DataFrame.
pub fn read_file(path: &Path, format: FileFormat) -> AppResult<DataFrame> {
    let df = match format {
        FileFormat::Csv | FileFormat::Tsv => {
            let separator = if format == FileFormat::Tsv {
                b'\t'
            } else {
                b','
            };
            CsvReadOptions::default()
                .with_has_header(true)
                .with_infer_schema_length(Some(10_000))
                .with_parse_options(
                    CsvParseOptions::default()
                        .with_separator(separator)
                        .with_try_parse_dates(false),
                )
                .try_into_reader_with_file_path(Some(path.to_path_buf()))?
                .finish()?
        }
        FileFormat::Json | FileFormat::NdJson => {
            let json_format = if format == FileFormat::Json && starts_with_array(path)? {
                JsonFormat::Json
            } else {
                JsonFormat::JsonLines
            };
            JsonReader::new(File::open(path)?)
                .with_json_format(json_format)
                .finish()?
        }
        FileFormat::Parquet => ParquetReader::new(File::open(path)?).finish()?,
    };
    if df.height() == 0 {
        return Err(AppError::bad_request("the uploaded file contains no rows"));
    }
    Ok(df)
}

fn starts_with_array(path: &Path) -> AppResult<bool> {
    let mut buf = [0u8; 512];
    let n = File::open(path)?.read(&mut buf)?;
    Ok(buf[..n].iter().find(|b| !b.is_ascii_whitespace()) == Some(&b'['))
}

pub fn write_parquet(df: &mut DataFrame, path: &Path) -> AppResult<()> {
    let file = File::create(path)?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(df)
        .map_err(|e| AppError::internal(format!("failed to write {}: {e}", path.display())))?;
    Ok(())
}

pub fn read_parquet(path: &Path) -> AppResult<DataFrame> {
    Ok(ParquetReader::new(File::open(path)?).finish()?)
}

/// Converts DataFrame rows into JSON objects (dates become ISO strings).
pub fn rows(df: &mut DataFrame) -> AppResult<Vec<Map<String, Value>>> {
    let mut buf = Vec::new();
    JsonWriter::new(&mut buf)
        .with_json_format(JsonFormat::Json)
        .finish(df)
        .map_err(|e| AppError::internal(format!("failed to convert rows: {e}")))?;
    let values: Vec<Value> = serde_json::from_slice(&buf)?;
    Ok(values
        .into_iter()
        .filter_map(|v| match v {
            Value::Object(map) => Some(map),
            _ => None,
        })
        .collect())
}

/// Reads a JSONL file of objects, skipping blank and malformed lines.
pub fn read_jsonl(path: &Path) -> AppResult<Vec<Map<String, Value>>> {
    let reader = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    for (n, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&line) {
            Ok(Value::Object(map)) => out.push(map),
            _ => {
                tracing::warn!(file = %path.display(), line = n + 1, "skipping malformed JSONL line")
            }
        }
    }
    Ok(out)
}

// ─── Items ───────────────────────────────────────────────────────────────────

/// Which columns identify and describe an item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemSchema {
    /// `None` means ids are derived from row content.
    pub id_column: Option<String>,
    pub text_columns: Vec<String>,
}

/// A column name and whether it holds text (strings or lists of strings).
pub type ColumnProfile = (String, bool);

pub fn profile_frame(df: &DataFrame) -> Vec<ColumnProfile> {
    df.schema()
        .iter()
        .map(|(name, dtype)| {
            let texty = matches!(dtype, DataType::String)
                || matches!(dtype, DataType::List(inner) if **inner == DataType::String);
            (name.to_string(), texty)
        })
        .collect()
}

pub fn profile_rows(rows: &[Map<String, Value>]) -> Vec<ColumnProfile> {
    let mut columns: Vec<ColumnProfile> = Vec::new();
    for row in rows.iter().take(1000) {
        for (key, value) in row {
            let texty = match value {
                Value::String(_) => Some(true),
                Value::Array(items) => Some(items.iter().all(Value::is_string)),
                Value::Null => None,
                _ => Some(false),
            };
            match columns.iter_mut().find(|(name, _)| name == key) {
                Some(entry) => {
                    if let Some(t) = texty {
                        entry.1 = entry.1 || t;
                    }
                }
                None => columns.push((key.clone(), texty.unwrap_or(false))),
            }
        }
    }
    columns
}

/// Resolves the configured (or auto-detected) id and text columns against the
/// columns present in the data.
///
/// With `strict` (file uploads), configured columns must exist in the data. JSON
/// item batches are not strict because individual objects may omit fields.
pub fn resolve_item_schema(
    columns: &[ColumnProfile],
    id_column: Option<&str>,
    text_columns: &[String],
    strict: bool,
) -> AppResult<ItemSchema> {
    let find = |wanted: &str| {
        columns
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(wanted))
            .map(|(n, t)| (n.clone(), *t))
    };
    let available = || {
        columns
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };

    if !strict && !text_columns.is_empty() {
        return Ok(ItemSchema {
            id_column: id_column.map(str::to_string),
            text_columns: text_columns.to_vec(),
        });
    }

    let id_column = match id_column {
        Some(wanted) if strict => Some(
            find(wanted)
                .ok_or_else(|| {
                    AppError::bad_request(format!(
                        "id column '{wanted}' not found; available columns: {}",
                        available()
                    ))
                })?
                .0,
        ),
        Some(wanted) => Some(find(wanted).map_or_else(|| wanted.to_string(), |(name, _)| name)),
        None => ID_CANDIDATES
            .iter()
            .find_map(|c| find(c))
            .map(|(name, _)| name),
    };

    let text_columns = if text_columns.is_empty() {
        let mut detected: Vec<String> = TEXT_CANDIDATES
            .iter()
            .filter_map(|c| find(c))
            .filter(|(name, texty)| *texty && Some(name) != id_column.as_ref())
            .map(|(name, _)| name)
            .collect();
        if detected.is_empty() {
            detected = columns
                .iter()
                .filter(|(name, texty)| *texty && Some(name) != id_column.as_ref())
                .map(|(name, _)| name.clone())
                .collect();
        }
        if detected.is_empty() {
            return Err(AppError::bad_request(format!(
                "no text column found to embed; pass text_columns explicitly. Available columns: {}",
                available()
            )));
        }
        detected
    } else {
        text_columns
            .iter()
            .map(|wanted| {
                find(wanted).map(|(name, _)| name).ok_or_else(|| {
                    AppError::bad_request(format!(
                        "text column '{wanted}' not found; available columns: {}",
                        available()
                    ))
                })
            })
            .collect::<AppResult<Vec<_>>>()?
    };

    Ok(ItemSchema {
        id_column,
        text_columns,
    })
}

/// Builds items from rows. Rows without an id or without any text are skipped
/// and counted.
pub fn build_items(rows: Vec<Map<String, Value>>, schema: &ItemSchema) -> (Vec<Item>, u64) {
    let mut items = Vec::with_capacity(rows.len());
    let mut skipped = 0;
    for row in rows {
        let id = match &schema.id_column {
            Some(column) => row.get(column).and_then(value_to_id),
            None => Some(content_id(&Value::Object(row.clone()).to_string())),
        };
        let text = schema
            .text_columns
            .iter()
            .filter_map(|c| row.get(c).and_then(value_to_text))
            .filter(|t| !t.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        match id {
            Some(id) if !text.trim().is_empty() && id.len() <= MAX_ENTITY_ID_LEN => {
                let text = match text.char_indices().nth(MAX_TEXT_CHARS) {
                    Some((cut, _)) => text[..cut].to_string(),
                    None => text,
                };
                items.push(Item {
                    id,
                    text,
                    payload: row,
                });
            }
            _ => skipped += 1,
        }
    }
    (items, skipped)
}

fn value_to_id(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn value_to_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().filter_map(value_to_text).collect();
            (!parts.is_empty()).then(|| parts.join(", "))
        }
        _ => None,
    }
}

// ─── Interactions ────────────────────────────────────────────────────────────

const USER_ALIASES: &[&str] = &[
    "user_id",
    "user",
    "userid",
    "customer_id",
    "customerid",
    "visitor_id",
    "session_id",
];
const ITEM_ALIASES: &[&str] = &[
    "item_id",
    "item",
    "itemid",
    "product_id",
    "productid",
    "sku",
    "doc_id",
];
const WEIGHT_ALIASES: &[&str] = &["weight", "rating", "score", "value", "strength"];
const EVENT_ALIASES: &[&str] = &["event_type", "event", "type", "action"];
const TIME_ALIASES: &[&str] = &[
    "timestamp",
    "ts",
    "time",
    "created_at",
    "event_time",
    "date",
];

/// Parses interaction rows, accepting common column aliases
/// (`user`/`customer_id`, `product_id`/`sku`, `rating`, ...).
pub fn parse_interactions(rows: Vec<Map<String, Value>>) -> AppResult<(Vec<Interaction>, u64)> {
    let keys: Vec<String> = profile_rows(&rows).into_iter().map(|(k, _)| k).collect();
    let pick = |aliases: &[&str]| {
        aliases
            .iter()
            .find_map(|a| keys.iter().find(|k| k.eq_ignore_ascii_case(a)))
            .cloned()
    };
    let user_col = pick(USER_ALIASES).ok_or_else(|| {
        AppError::bad_request(format!(
            "interactions need a user column (one of: {})",
            USER_ALIASES.join(", ")
        ))
    })?;
    let item_col = pick(ITEM_ALIASES).ok_or_else(|| {
        AppError::bad_request(format!(
            "interactions need an item column (one of: {})",
            ITEM_ALIASES.join(", ")
        ))
    })?;
    let weight_col = pick(WEIGHT_ALIASES);
    let event_col = pick(EVENT_ALIASES);
    let time_col = pick(TIME_ALIASES);

    let mut out = Vec::with_capacity(rows.len());
    let mut skipped = 0;
    for row in rows {
        let (Some(user_id), Some(item_id)) = (
            row.get(&user_col).and_then(value_to_id),
            row.get(&item_col).and_then(value_to_id),
        ) else {
            skipped += 1;
            continue;
        };
        if user_id.len() > MAX_ENTITY_ID_LEN || item_id.len() > MAX_ENTITY_ID_LEN {
            skipped += 1;
            continue;
        }
        out.push(Interaction {
            user_id,
            item_id,
            weight: weight_col
                .as_ref()
                .and_then(|c| row.get(c))
                .and_then(value_to_f32),
            event_type: event_col
                .as_ref()
                .and_then(|c| row.get(c))
                .and_then(|v| v.as_str().map(str::to_string)),
            timestamp: time_col
                .as_ref()
                .and_then(|c| row.get(c))
                .and_then(value_to_time),
            prediction_id: row
                .get("prediction_id")
                .and_then(|v| v.as_str().map(str::to_string)),
        });
    }
    Ok((out, skipped))
}

fn value_to_f32(v: &Value) -> Option<f32> {
    match v {
        Value::Number(n) => n.as_f64().map(|x| x as f32),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
    .filter(|x: &f32| x.is_finite())
}

fn value_to_time(v: &Value) -> Option<DateTime<Utc>> {
    match v {
        Value::String(s) => DateTime::parse_from_rfc3339(s)
            .map(|t| t.with_timezone(&Utc))
            .ok()
            .or_else(|| {
                chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                    .ok()
                    .map(|t| t.and_utc())
            }),
        Value::Number(n) => {
            let secs = n.as_f64()?;
            // Heuristic: anything above 1e12 is already in milliseconds.
            let millis = if secs > 1e12 { secs } else { secs * 1000.0 };
            Utc.timestamp_millis_opt(millis as i64).single()
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn write(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::File::create(&path)
            .unwrap()
            .write_all(content.as_bytes())
            .unwrap();
        path
    }

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn detects_formats() {
        assert_eq!(FileFormat::from_filename("a.CSV"), Some(FileFormat::Csv));
        assert_eq!(
            FileFormat::from_filename("a.jsonl"),
            Some(FileFormat::NdJson)
        );
        assert_eq!(
            FileFormat::from_filename("x.y.parquet"),
            Some(FileFormat::Parquet)
        );
        assert_eq!(FileFormat::from_filename("a.txt"), None);
        assert_eq!(FileFormat::from_filename("noext"), None);
    }

    #[test]
    fn reads_csv_json_array_ndjson_and_parquet() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write(dir.path(), "a.csv", "id,name,price\n1,Shoe,9.5\n2,Hat,3\n");
        let mut df = read_file(&csv, FileFormat::Csv).unwrap();
        assert_eq!(df.height(), 2);
        let r = rows(&mut df).unwrap();
        assert_eq!(r[0]["name"], "Shoe");
        assert_eq!(r[0]["id"], 1);

        let arr = write(
            dir.path(),
            "a.json",
            r#"  [{"id": "a", "tags": ["x", "y"]}, {"id": "b", "tags": []}]"#,
        );
        assert_eq!(read_file(&arr, FileFormat::Json).unwrap().height(), 2);

        let nd = write(
            dir.path(),
            "b.json",
            "{\"id\": 1}\n{\"id\": 2}\n{\"id\": 3}\n",
        );
        assert_eq!(read_file(&nd, FileFormat::Json).unwrap().height(), 3);

        let pq = dir.path().join("a.parquet");
        write_parquet(&mut df, &pq).unwrap();
        assert_eq!(read_file(&pq, FileFormat::Parquet).unwrap().height(), 2);

        let empty = write(dir.path(), "e.csv", "id,name\n");
        assert!(read_file(&empty, FileFormat::Csv).is_err());
    }

    #[test]
    fn auto_detects_item_schema() {
        let cols = vec![
            ("sku".into(), true),
            ("description".into(), true),
            ("Name".into(), true),
            ("price".into(), false),
        ];
        let schema = resolve_item_schema(&cols, None, &[], true).unwrap();
        assert_eq!(schema.id_column.as_deref(), Some("sku"));
        assert_eq!(schema.text_columns, vec!["Name", "description"]);

        let explicit =
            resolve_item_schema(&cols, Some("SKU"), &["description".into()], true).unwrap();
        assert_eq!(explicit.id_column.as_deref(), Some("sku"));
        assert_eq!(explicit.text_columns, vec!["description"]);

        assert!(resolve_item_schema(&cols, Some("missing"), &[], true).is_err());
        assert!(resolve_item_schema(&cols, None, &["missing".into()], true).is_err());
        let lenient = resolve_item_schema(&cols, Some("id"), &["missing".into()], false).unwrap();
        assert_eq!(lenient.text_columns, vec!["missing"]);
        assert!(
            resolve_item_schema(
                &[("id".into(), false), ("n".into(), false)],
                None,
                &[],
                true
            )
            .is_err()
        );
    }

    #[test]
    fn falls_back_to_any_text_column_and_content_ids() {
        let cols = vec![("blurb".into(), true), ("count".into(), false)];
        let schema = resolve_item_schema(&cols, None, &[], true).unwrap();
        assert_eq!(schema.id_column, None);
        assert_eq!(schema.text_columns, vec!["blurb"]);

        let row = obj(json!({"blurb": "hello", "count": 1}));
        let (a, _) = build_items(vec![row.clone()], &schema);
        let (b, _) = build_items(vec![row], &schema);
        assert_eq!(a[0].id, b[0].id, "content ids are stable");
    }

    #[test]
    fn builds_items_and_skips_bad_rows() {
        let schema = ItemSchema {
            id_column: Some("id".into()),
            text_columns: vec!["name".into(), "tags".into()],
        };
        let (items, skipped) = build_items(
            vec![
                obj(json!({"id": 1, "name": "Shoe", "tags": ["run", "sport"]})),
                obj(json!({"id": null, "name": "No id"})),
                obj(json!({"id": "3", "name": ""})),
            ],
            &schema,
        );
        assert_eq!(skipped, 2);
        assert_eq!(items[0].id, "1");
        assert_eq!(items[0].text, "Shoe\nrun, sport");
        assert_eq!(items[0].payload["tags"][1], "sport");
    }

    #[test]
    fn parses_interactions_with_aliases() {
        let (parsed, skipped) = parse_interactions(vec![
            obj(json!({"customer_id": 7, "SKU": "a", "rating": "4.5", "ts": 1_700_000_000})),
            obj(json!({"customer_id": "8", "SKU": "b", "rating": 2, "ts": "2026-01-02T03:04:05Z"})),
            obj(json!({"customer_id": null, "SKU": "c"})),
        ])
        .unwrap();
        assert_eq!(skipped, 1);
        assert_eq!(parsed[0].user_id, "7");
        assert_eq!(parsed[0].weight, Some(4.5));
        assert_eq!(parsed[0].timestamp.unwrap().timestamp(), 1_700_000_000);
        assert_eq!(
            parsed[1].timestamp.unwrap().to_rfc3339(),
            "2026-01-02T03:04:05+00:00"
        );

        assert!(parse_interactions(vec![obj(json!({"who": 1, "what": 2}))]).is_err());
    }
}
