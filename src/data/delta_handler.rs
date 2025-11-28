use anyhow::{Context, Result};
use deltalake::arrow::array::RecordBatch;
use deltalake::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use deltalake::*;

use polars::prelude::*;

use serde_json::Value;
use bytes::Bytes;
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;

pub struct DeltaTableHandler {
    table_path: String,
}

impl DeltaTableHandler {
    pub async fn new() -> Self {
        let table_path = std::env::var("DELTA_TABLE_PATH")
            .unwrap_or_else(|_| "/data/delta/tables".to_string());
            
        // Ensure the delta table exists
        let _ = deltalake::open_table(&table_path).await
            .or_else(|_| {
                deltalake::DeltaTableBuilder::from_uri(&table_path)
                .build()
            });
            
        Self { table_path }
    }
    
    pub async fn process_file(&self, file_path: &str) -> Result<String> {
        let dataset_id = Uuid::new_v4().to_string();

        // Read the file based on extension
        let df = match file_path.split('.').last() {
            Some("csv") => LazyCsvReader::new(PlPath::from_str(file_path)).finish(),
            Some("parquet") => LazyFrame::scan_parquet(PlPath::from_str(file_path),Default::default()),
            Some("json") => LazyJsonLineReader::new(PlPath::from_str(file_path)).finish(),
            _ => anyhow::bail!("Unsupported file format: {}", file_path),
        }?
        .with_columns([
            lit(dataset_id.clone()).alias("dataset_id"),
            lit(Utc::now().to_rfc3339()).alias("processed_at"),
        ]);

        // Collect the DataFrame
        let collected_df = df.collect().context("Failed to collect DataFrame")?;

        // Convert Polars DataFrame to Arrow record batches via Parquet serialization
        let batches = polars_to_arrow_batches(collected_df)
            .context("Failed to convert Polars DataFrame to Arrow RecordBatches")?;

        // Write to Delta table
        let table = deltalake::open_table(&self.table_path)
            .await
            .context("Failed to open Delta table")?;

        DeltaOps(table)
            .write(batches)
            .with_save_mode(deltalake::protocol::SaveMode::Append)
            .await
            .context("Failed to write to Delta table")?;

        Ok(dataset_id)
    }
    
    pub async fn list_datasets(&self) -> Result<Vec<HashMap<String, Value>>> {
        let table = open_table(&self.table_path).await?;
        let uris: Vec<String> = table.get_file_uris()?.collect();

        // build whatever JSON you need; here we just keep the URI
        Ok(uris
            .into_iter()
            .map(|uri| {
                let mut m = HashMap::new();
                m.insert("file_path".to_string(), Value::String(uri));
                m
            })
            .collect())
    }
    
    pub async fn get_dataset_info(&self, dataset_id: &str) -> Result<HashMap<String, Value>> {
        //let table = deltalake::open_table(&self.table_path).await?;
        
        // This is a simplified implementation
        // In a real implementation, you would query the dataset metadata
        let mut info = HashMap::new();
        info.insert("dataset_id".to_string(), Value::String(dataset_id.to_string()));
        info.insert("row_count".to_string(), Value::Number(1000.into())); // Placeholder
        info.insert("columns".to_string(), Value::Array(vec![
            Value::String("id".to_string()),
            Value::String("text".to_string()),
            Value::String("metadata".to_string()),
        ]));
        
        Ok(info)
    }
}

/// Convert a Polars DataFrame to standard Arrow RecordBatches via Parquet serialization.
/// This bridges polars-arrow and the standard arrow crate used by Delta Lake.
fn polars_to_arrow_batches(mut df: DataFrame) -> Result<Vec<RecordBatch>> {
    // Write DataFrame to Parquet in memory
    let mut buffer = Vec::new();
    ParquetWriter::new(&mut buffer)
        .finish(&mut df)
        .context("Failed to write DataFrame to Parquet buffer")?;

    // Read back using standard arrow-parquet reader (Bytes implements ChunkReader)
    let bytes = Bytes::from(buffer);
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .context("Failed to create Parquet reader")?
        .build()
        .context("Failed to build Parquet record batch reader")?;

    let batches: Vec<RecordBatch> = reader
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("Failed to read Arrow record batches from Parquet")?;

    Ok(batches)
}