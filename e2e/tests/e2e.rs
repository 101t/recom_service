//! Black-box end-to-end tests for a running recommendation service.
//!
//! They talk HTTP only, so they work against any deployment:
//!
//! ```text
//! make e2e                                   # builds the image, starts Qdrant + service, runs these tests
//! E2E_BASE_URL=http://host:8000 cargo test -p recom-e2e
//! ```
//!
//! Environment:
//! * `E2E_BASE_URL` — service URL (default `http://localhost:8000`)
//! * `E2E_API_KEY` — sent as `X-API-Key` when the service requires one
//! * `E2E_STARTUP_TIMEOUT_SECS` — how long to wait for `/ready` (default 300; the
//!   first start downloads the embedding model)
//!
//! Every test works on its own uniquely named dataset and deletes it at the end,
//! so tests run in parallel and can be pointed at shared environments.

use std::time::{Duration, Instant};

use reqwest::{Client, StatusCode, multipart};
use serde_json::{Value, json};

const PRODUCTS_CSV: &[u8] = include_bytes!("../data/products.csv");
const ARTICLES_JSON: &[u8] = include_bytes!("../data/articles.json");
const INTERACTIONS_CSV: &[u8] = include_bytes!("../data/interactions.csv");
const MULTILINGUAL_JSON: &[u8] = include_bytes!("../data/multilingual.json");

struct Api {
    http: Client,
    base: String,
    key: Option<String>,
}

impl Api {
    /// Connects and waits until the service reports ready.
    async fn connect() -> Self {
        let base = std::env::var("E2E_BASE_URL").unwrap_or_else(|_| "http://localhost:8000".into());
        let api = Self {
            http: Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .unwrap(),
            base: base.trim_end_matches('/').to_string(),
            key: std::env::var("E2E_API_KEY").ok().filter(|k| !k.is_empty()),
        };
        let timeout = std::env::var("E2E_STARTUP_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);
        let deadline = Instant::now() + Duration::from_secs(timeout);
        loop {
            match api.http.get(api.url("/ready")).send().await {
                Ok(resp) if resp.status().is_success() => return api,
                _ if Instant::now() < deadline => tokio::time::sleep(Duration::from_secs(2)).await,
                other => panic!("service at {} is not ready: {other:?}", api.base),
            }
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let builder = self.http.request(method, self.url(path));
        match &self.key {
            Some(key) => builder.header("x-api-key", key),
            None => builder,
        }
    }

    async fn send(&self, builder: reqwest::RequestBuilder) -> (StatusCode, Value) {
        let resp = builder.send().await.expect("request failed");
        let status = resp.status();
        let text = resp.text().await.unwrap();
        (
            status,
            serde_json::from_str(&text).unwrap_or(Value::String(text)),
        )
    }

    async fn get(&self, path: &str) -> (StatusCode, Value) {
        self.send(self.request(reqwest::Method::GET, path)).await
    }

    async fn post(&self, path: &str, body: Value) -> (StatusCode, Value) {
        self.send(self.request(reqwest::Method::POST, path).json(&body))
            .await
    }

    async fn delete(&self, path: &str) -> StatusCode {
        self.request(reqwest::Method::DELETE, path)
            .send()
            .await
            .unwrap()
            .status()
    }

    async fn upload(&self, path: &str, filename: &str, content: &[u8]) -> (StatusCode, Value) {
        let part = multipart::Part::bytes(content.to_vec()).file_name(filename.to_string());
        self.send(
            self.request(reqwest::Method::POST, path)
                .multipart(multipart::Form::new().part("file", part)),
        )
        .await
    }

    /// Uploads a catalog file, waiting for indexing to finish.
    async fn load(&self, dataset: &str, filename: &str, content: &[u8]) -> Value {
        let (status, body) = self
            .upload(
                &format!("/api/v1/datasets/{dataset}/items/upload?wait=true"),
                filename,
                content,
            )
            .await;
        assert_eq!(status, StatusCode::OK, "upload failed: {body}");
        assert_eq!(body["dataset"]["status"], "ready", "{body}");
        body
    }

    async fn recommend(&self, dataset: &str, body: Value) -> Value {
        let (status, response) = self
            .post(
                &format!("/api/v1/datasets/{dataset}/recommend"),
                body.clone(),
            )
            .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "recommend {body} failed: {response}"
        );
        response
    }

    async fn wait_ready(&self, dataset: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let (_, body) = self.get(&format!("/api/v1/datasets/{dataset}")).await;
            match body["status"].as_str() {
                Some("ready") => return body,
                Some("failed") => panic!("training failed: {body}"),
                _ if Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(250)).await
                }
                _ => panic!("dataset {dataset} never became ready: {body}"),
            }
        }
    }

    async fn cleanup(&self, dataset: &str) {
        self.delete(&format!("/api/v1/datasets/{dataset}")).await;
    }
}

fn unique(prefix: &str) -> String {
    format!(
        "{prefix}-{}",
        &uuid::Uuid::new_v4().simple().to_string()[..10]
    )
}

fn ids(response: &Value) -> Vec<String> {
    response["recommendations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_string())
        .collect()
}

fn top(response: &Value, n: usize) -> Vec<String> {
    ids(response).into_iter().take(n).collect()
}

#[tokio::test]
async fn service_is_healthy_and_documented() {
    let api = Api::connect().await;

    let (status, health) = api.get("/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["status"], "ok");

    let (status, info) = api.get("/api/v1/info").await;
    assert_eq!(status, StatusCode::OK, "{info}");
    assert!(info["embedding"]["dimension"].as_u64().unwrap() > 0);
    assert!(
        info["vector_store"]["collections"]
            .as_array()
            .unwrap()
            .len()
            >= 2
    );

    let (status, openapi) = api.get("/api-docs/openapi.json").await;
    assert_eq!(status, StatusCode::OK);
    assert!(openapi["paths"]["/api/v1/datasets/{dataset_id}/recommend"].is_object());

    let metrics = api.http.get(api.url("/metrics")).send().await.unwrap();
    assert_eq!(metrics.status(), StatusCode::OK);
    assert!(
        metrics
            .text()
            .await
            .unwrap()
            .contains("recom_http_requests_total")
    );
}

#[tokio::test]
async fn catalog_upload_semantic_search_and_filters() {
    let api = Api::connect().await;
    let ds = unique("products");
    let body = api.load(&ds, "products.csv", PRODUCTS_CSV).await;
    assert_eq!(body["dataset"]["stats"]["items"], 20);

    // Semantic match without keyword overlap ("jogging" vs "running").
    let res = api
        .recommend(
            &ds,
            json!({"query": "comfortable sneakers for jogging", "limit": 5}),
        )
        .await;
    assert!(top(&res, 3).contains(&"1".to_string()), "{res}");

    let res = api
        .recommend(
            &ds,
            json!({
                "query": "gadgets",
                "limit": 10,
                "filter": {
                    "must": [{"key": "category", "match": {"value": "electronics"}}],
                    "must_not": [{"key": "in_stock", "match": {"value": false}}]
                }
            }),
        )
        .await;
    let recs = res["recommendations"].as_array().unwrap();
    assert!(!recs.is_empty());
    assert!(
        recs.iter()
            .all(|r| r["payload"]["category"] == "electronics" && r["payload"]["in_stock"] == true)
    );

    let res = api
        .recommend(&ds, json!({"query": "jacket", "filter": {"must": [{"key": "price", "range": {"lt": 250}}]}}))
        .await;
    assert!(
        res["recommendations"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["payload"]["price"].as_f64().unwrap() < 250.0)
    );

    // Item-to-item: more like the running shoes, never the shoes themselves.
    let res = api
        .recommend(&ds, json!({"item_ids": ["1"], "limit": 3}))
        .await;
    assert!(!ids(&res).contains(&"1".to_string()));
    assert!(
        res["recommendations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["payload"]["category"] == "footwear"),
        "{res}"
    );

    // Pagination and diversity.
    let page = api
        .recommend(&ds, json!({"query": "outdoor equipment", "limit": 4}))
        .await;
    let second = api
        .recommend(
            &ds,
            json!({"query": "outdoor equipment", "limit": 2, "offset": 2}),
        )
        .await;
    assert_eq!(ids(&second), ids(&page)[2..]);
    let diverse = api
        .recommend(
            &ds,
            json!({"query": "outdoor equipment", "limit": 5, "diversity": 0.8}),
        )
        .await;
    assert_eq!(ids(&diverse).len(), 5);

    api.cleanup(&ds).await;
}

#[tokio::test]
async fn multilingual_queries_find_english_items() {
    let api = Api::connect().await;
    let ds = unique("multi-q");
    api.load(&ds, "products.csv", PRODUCTS_CSV).await;

    for (query, expected) in [
        ("zapatillas para correr", "1"),      // Spanish: running shoes
        ("سماعات لاسلكية عازلة للضوضاء", "4"), // Arabic: noise-cancelling wireless headphones
        ("veste en cuir pour moto", "3"),     // French: leather motorcycle jacket
        ("isolierte Trinkflasche", "6"),      // German: insulated water bottle
        ("防滑瑜伽垫", "5"),                  // Chinese: non-slip yoga mat
    ] {
        let res = api
            .recommend(
                &ds,
                json!({"query": query, "limit": 3, "with_payload": false}),
            )
            .await;
        assert!(
            top(&res, 3).contains(&expected.to_string()),
            "query '{query}' expected item {expected}: {res}"
        );
    }
    api.cleanup(&ds).await;
}

#[tokio::test]
async fn english_queries_find_multilingual_items() {
    let api = Api::connect().await;
    let ds = unique("multi-items");
    let body = api.load(&ds, "multilingual.json", MULTILINGUAL_JSON).await;
    assert_eq!(body["dataset"]["stats"]["items"], 8);

    for (query, expected) in [
        ("lightweight running shoes", "m1"),
        ("noise cancelling headphones", "m2"),
        ("leather motorcycle jacket", "m3"),
        ("waterproof hiking boots", "m4"),
    ] {
        let res = api
            .recommend(&ds, json!({"query": query, "limit": 2}))
            .await;
        assert!(
            top(&res, 2).contains(&expected.to_string()),
            "query '{query}' expected {expected}: {res}"
        );
    }
    api.cleanup(&ds).await;
}

#[tokio::test]
async fn interactions_personalize_recommendations() {
    let api = Api::connect().await;
    let ds = unique("personal");
    api.load(&ds, "products.csv", PRODUCTS_CSV).await;

    // Bulk history upload.
    let (status, body) = api
        .upload(
            &format!("/api/v1/datasets/{ds}/interactions/upload?wait=true"),
            "events.csv",
            INTERACTIONS_CSV,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["dataset"]["stats"]["users"], 3);

    // u4 bought/viewed outerwear (3, 8, 11): recommend unseen outerwear first.
    let res = api
        .recommend(&ds, json!({"user_id": "u4", "limit": 5}))
        .await;
    let recs = ids(&res);
    assert!(
        recs.iter()
            .all(|id| !["3", "8", "11"].contains(&id.as_str())),
        "seen items excluded: {recs:?}"
    );
    assert!(
        recs.contains(&"17".to_string()),
        "expected the remaining outerwear item: {res}"
    );

    // Real-time events for a new user.
    let (status, body) = api
        .post(
            &format!("/api/v1/datasets/{ds}/interactions"),
            json!({"interactions": [
                {"user_id": "fresh", "item_id": "4", "event_type": "purchase", "weight": 2},
                {"user_id": "fresh", "item_id": "12", "event_type": "view"}
            ]}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["accepted"], 2);
    let res = api
        .recommend(&ds, json!({"user_id": "fresh", "limit": 3}))
        .await;
    let electronics = res["recommendations"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["payload"]["category"] == "electronics")
        .count();
    assert!(electronics >= 2, "audio fan should get electronics: {res}");

    // Personalization blended with a text query and a filter.
    let res = api
        .recommend(
            &ds,
            json!({
                "user_id": "fresh",
                "query": "something for the gym",
                "filter": {"must": [{"key": "category", "match": {"value": "fitness"}}]},
                "limit": 3
            }),
        )
        .await;
    assert!(
        res["recommendations"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["payload"]["category"] == "fitness")
    );

    let (status, _) = api
        .post(
            &format!("/api/v1/datasets/{ds}/recommend"),
            json!({"user_id": "stranger"}),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    api.cleanup(&ds).await;
}

#[tokio::test]
async fn json_items_background_upload_and_retrain() {
    let api = Api::connect().await;
    let ds = unique("articles");

    let (status, created) = api
        .post(
            "/api/v1/datasets",
            json!({
                "id": ds, "name": "Articles", "id_column": "id", "text_columns": ["title", "text"]
            }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(created["status"], "pending");

    // Background upload: 202 now, ready later.
    let (status, accepted) = api
        .upload(
            &format!("/api/v1/datasets/{ds}/items/upload"),
            "articles.json",
            ARTICLES_JSON,
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    let ready = api.wait_ready(&ds).await;
    assert_eq!(ready["stats"]["items"], 10);

    // JSON upsert, read and delete.
    let (status, body) = api
        .post(&format!("/api/v1/datasets/{ds}/items"), json!({"items": [
            {"id": 111, "title": "Vector Databases Explained", "text": "How approximate nearest neighbour search powers semantic retrieval"}
        ]}))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["upserted"], 1);
    let res = api
        .recommend(
            &ds,
            json!({"query": "nearest neighbour search engines", "limit": 3}),
        )
        .await;
    assert_eq!(top(&res, 1), vec!["111"], "{res}");
    let (status, item) = api.get(&format!("/api/v1/datasets/{ds}/items/111")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(item["payload"]["title"], "Vector Databases Explained");
    assert_eq!(
        api.delete(&format!("/api/v1/datasets/{ds}/items/105"))
            .await,
        StatusCode::NO_CONTENT
    );

    // Retrain from stored data reproduces the index (10 + 1 - 1 items).
    let (status, trained) = api
        .post(
            &format!("/api/v1/datasets/{ds}/train?wait=true&reset=true"),
            json!({}),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{trained}");
    assert_eq!(trained["dataset"]["stats"]["items"], 10);
    assert_eq!(
        api.get(&format!("/api/v1/datasets/{ds}/items/105")).await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        api.get(&format!("/api/v1/datasets/{ds}/items/111")).await.0,
        StatusCode::OK
    );

    api.cleanup(&ds).await;
    assert_eq!(
        api.get(&format!("/api/v1/datasets/{ds}")).await.0,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn predictions_are_logged_and_exportable() {
    let api = Api::connect().await;
    let ds = unique("logs");
    api.load(&ds, "products.csv", PRODUCTS_CSV).await;
    let res = api.recommend(&ds, json!({"query": "water bottle"})).await;
    let prediction_id = res["prediction_id"].as_str().unwrap().to_string();

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (_, list) = api
            .get(&format!("/api/v1/predictions?dataset_id={ds}"))
            .await;
        if list["logs"]
            .as_array()
            .is_some_and(|logs| logs.iter().any(|l| l["prediction_id"] == prediction_id))
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "prediction {prediction_id} was not logged"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    let resp = api
        .request(
            reqwest::Method::GET,
            &format!("/api/v1/predictions/export?format=csv&dataset_id={ds}"),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let csv = resp.text().await.unwrap();
    assert!(csv.starts_with("prediction_id,") && csv.contains(&prediction_id));
    api.cleanup(&ds).await;
}

#[tokio::test]
async fn errors_use_a_consistent_envelope() {
    let api = Api::connect().await;
    let ds = unique("errors");
    api.load(&ds, "products.csv", PRODUCTS_CSV).await;

    let cases = [
        (
            api.post(&format!("/api/v1/datasets/{ds}/recommend"), json!({}))
                .await,
            StatusCode::BAD_REQUEST,
        ),
        (
            api.post(
                &format!("/api/v1/datasets/{ds}/recommend"),
                json!({"query": "x", "limit": 100000}),
            )
            .await,
            StatusCode::BAD_REQUEST,
        ),
        (
            api.post(
                &format!("/api/v1/datasets/{ds}/recommend"),
                json!({"item_ids": ["404"]}),
            )
            .await,
            StatusCode::NOT_FOUND,
        ),
        (
            api.post(
                "/api/v1/datasets/does-not-exist/recommend",
                json!({"query": "x"}),
            )
            .await,
            StatusCode::NOT_FOUND,
        ),
        (
            api.upload(
                &format!("/api/v1/datasets/{ds}/items/upload"),
                "notes.txt",
                b"hi",
            )
            .await,
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
        ),
    ];
    for ((status, body), expected) in cases {
        assert_eq!(status, expected, "{body}");
        assert!(
            body["error"]["code"].is_string() && body["error"]["message"].is_string(),
            "{body}"
        );
    }
    api.cleanup(&ds).await;
}
