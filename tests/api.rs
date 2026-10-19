//! API tests through the full HTTP stack (routing, middleware, services) using the
//! in-memory vector store and the hash embedder: hermetic, fast, no network.
//!
//! The black-box suite against a real Qdrant + multilingual model lives in `e2e/`.

use std::time::Duration;

use actix_http::Request;
use actix_web::body::MessageBody;
use actix_web::dev::{Service, ServiceResponse};
use actix_web::http::StatusCode;
use actix_web::test::{self, TestRequest};
use recom_service::config::Settings;
use recom_service::server::{build_app, metrics};
use recom_service::state::AppState;
use recom_service::storage::tabular::{self, FileFormat};
use serde_json::{Value, json};

const PRODUCTS_CSV: &str = include_str!("../e2e/data/products.csv");
const ARTICLES_JSON: &str = include_str!("../e2e/data/articles.json");
const INTERACTIONS_CSV: &str = include_str!("../e2e/data/interactions.csv");

async fn state(configure: impl FnOnce(&mut Settings)) -> (AppState, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut settings = Settings::for_testing(dir.path());
    configure(&mut settings);
    (AppState::initialize(settings).await.unwrap(), dir)
}

macro_rules! app {
    ($state:expr) => {
        test::init_service(build_app($state.clone(), metrics().unwrap())).await
    };
}

async fn call<S, B>(app: &S, req: Request) -> (StatusCode, Value)
where
    S: Service<Request, Response = ServiceResponse<B>, Error = actix_web::Error>,
    B: MessageBody,
{
    let resp = test::call_service(app, req).await;
    let status = resp.status();
    let body = test::read_body(resp).await;
    let json = serde_json::from_slice(&body)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&body).into()));
    (status, json)
}

fn get(uri: &str) -> Request {
    TestRequest::get().uri(uri).to_request()
}

fn post(uri: &str, body: Value) -> Request {
    TestRequest::post().uri(uri).set_json(body).to_request()
}

fn upload(uri: &str, filename: &str, content: &[u8]) -> Request {
    let boundary = "recom-test-boundary";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(content);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    TestRequest::post()
        .uri(uri)
        .insert_header((
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        ))
        .set_payload(body)
        .to_request()
}

fn ids(body: &Value) -> Vec<String> {
    body["recommendations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_string())
        .collect()
}

async fn load_products<S, B>(app: &S, dataset: &str) -> Value
where
    S: Service<Request, Response = ServiceResponse<B>, Error = actix_web::Error>,
    B: MessageBody,
{
    let (status, body) = call(
        app,
        upload(
            &format!("/api/v1/datasets/{dataset}/items/upload?wait=true"),
            "products.csv",
            PRODUCTS_CSV.as_bytes(),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

async fn wait_ready<S, B>(app: &S, dataset: &str) -> Value
where
    S: Service<Request, Response = ServiceResponse<B>, Error = actix_web::Error>,
    B: MessageBody,
{
    for _ in 0..200 {
        let (_, body) = call(app, get(&format!("/api/v1/datasets/{dataset}"))).await;
        match body["status"].as_str() {
            Some("ready") => return body,
            Some("failed") => panic!("training failed: {body}"),
            _ => actix_web::rt::time::sleep(Duration::from_millis(25)).await,
        }
    }
    panic!("dataset {dataset} never became ready");
}

#[actix_web::test]
async fn health_info_models_and_openapi() {
    let (state, _dir) = state(|_| {}).await;
    let app = app!(state);

    let (status, body) = call(&app, get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(call(&app, get("/ready")).await.0, StatusCode::OK);

    let (status, info) = call(&app, get("/api/v1/info")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(info["embedding"]["provider"], "hash");
    assert_eq!(info["embedding"]["dimension"], 384);
    assert_eq!(info["vector_store"]["kind"], "memory");

    let (status, models) = call(&app, get("/api/v1/models")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        models["models"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["model"] == "intfloat/multilingual-e5-small")
    );

    let (status, openapi) = call(&app, get("/api-docs/openapi.json")).await;
    assert_eq!(status, StatusCode::OK);
    for path in [
        "/health",
        "/api/v1/datasets",
        "/api/v1/datasets/{dataset_id}/recommend",
        "/api/v1/predictions/export",
    ] {
        assert!(
            openapi["paths"].get(path).is_some(),
            "missing {path} in OpenAPI"
        );
    }

    let (status, body) = call(&app, get("/api/v1/nope")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");

    let docs = test::call_service(&app, get("/docs/")).await;
    assert_eq!(
        docs.status(),
        StatusCode::OK,
        "Swagger UI is served from vendored assets"
    );

    let (status, metrics) = call(&app, get("/metrics")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        metrics
            .as_str()
            .unwrap()
            .contains("recom_http_requests_total")
    );
}

#[actix_web::test]
async fn upload_csv_then_recommend_with_filters() {
    let (state, _dir) = state(|_| {}).await;
    let app = app!(state);

    let body = load_products(&app, "products").await;
    let dataset = &body["dataset"];
    assert_eq!(body["rows"], 20);
    assert_eq!(dataset["status"], "ready");
    assert_eq!(dataset["stats"]["items"], 20);
    assert_eq!(dataset["id_column"], "id");
    assert_eq!(
        dataset["text_columns"],
        json!(["name", "description", "category"])
    );

    let uri = "/api/v1/datasets/products/recommend";
    let (status, body) = call(
        &app,
        post(uri, json!({"query": "running shoes", "limit": 5})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(ids(&body)[0], "1");
    assert_eq!(
        body["recommendations"][0]["payload"]["name"],
        "Running Shoes Pro"
    );
    assert!(
        body["prediction_id"]
            .as_str()
            .is_some_and(|p| !p.is_empty())
    );

    // Keyword match, numeric range, boolean exclusion.
    let (_, body) = call(
        &app,
        post(
            uri,
            json!({
                "query": "outdoor gear",
                "limit": 20,
                "filter": {
                    "must": [{"key": "category", "match": {"any": ["electronics", "fitness"]}}],
                    "must_not": [{"key": "in_stock", "match": {"value": false}}]
                }
            }),
        ),
    )
    .await;
    let recs = body["recommendations"].as_array().unwrap();
    assert!(!recs.is_empty());
    for r in recs {
        let p = &r["payload"];
        assert!(
            p["category"] == "electronics" || p["category"] == "fitness",
            "{p}"
        );
        assert_eq!(p["in_stock"], true);
    }
    let (_, body) = call(&app, post(uri, json!({
        "query": "jacket", "limit": 20, "filter": {"must": [{"key": "price", "range": {"lte": 100}}]}
    })))
    .await;
    assert!(
        body["recommendations"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["payload"]["price"].as_f64().unwrap() <= 100.0)
    );

    // Pagination, payload toggle, threshold and diversity.
    let (_, top2) = call(
        &app,
        post(uri, json!({"query": "running shoes", "limit": 2})),
    )
    .await;
    let (_, second) = call(
        &app,
        post(
            uri,
            json!({"query": "running shoes", "limit": 1, "offset": 1}),
        ),
    )
    .await;
    assert_eq!(ids(&second), ids(&top2)[1..]);
    let (_, bare) = call(
        &app,
        post(
            uri,
            json!({"query": "running shoes", "limit": 3, "with_payload": false}),
        ),
    )
    .await;
    assert!(bare["recommendations"][0].get("payload").is_none());
    let (_, strict) = call(
        &app,
        post(
            uri,
            json!({"query": "running shoes", "score_threshold": 0.4}),
        ),
    )
    .await;
    assert!(
        strict["recommendations"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["score"].as_f64().unwrap() >= 0.4)
    );
    let (_, diverse) = call(
        &app,
        post(
            uri,
            json!({"query": "running shoes", "limit": 5, "diversity": 0.9}),
        ),
    )
    .await;
    let mut diverse_ids = ids(&diverse);
    assert_eq!(diverse_ids.len(), 5);
    diverse_ids.dedup();
    assert_eq!(diverse_ids.len(), 5);
}

#[actix_web::test]
async fn item_to_item_recommendations() {
    let (state, _dir) = state(|_| {}).await;
    let app = app!(state);
    load_products(&app, "shop").await;
    let uri = "/api/v1/datasets/shop/recommend";

    let (status, body) = call(&app, post(uri, json!({"item_ids": ["1"], "limit": 5}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        !ids(&body).contains(&"1".to_string()),
        "example items are excluded"
    );

    let (status, body) = call(&app, post(uri, json!({
        "item_ids": ["1", "16"], "negative_item_ids": ["4"], "strategy": "best_score", "limit": 5
    })))
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let result = ids(&body);
    assert!(
        result
            .iter()
            .all(|id| !["1", "16", "4"].contains(&id.as_str()))
    );

    let (status, body) = call(&app, post(uri, json!({"item_ids": ["does-not-exist"]}))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("does-not-exist")
    );
}

#[actix_web::test]
async fn json_items_crud_and_background_upload() {
    let (state, _dir) = state(|_| {}).await;
    let app = app!(state);
    let articles: Value = serde_json::from_str(ARTICLES_JSON).unwrap();

    // JSON upsert auto-creates the dataset.
    let (status, body) = call(
        &app,
        post(
            "/api/v1/datasets/articles/items",
            json!({"items": articles}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["upserted"], 10);
    let (_, dataset) = call(&app, get("/api/v1/datasets/articles")).await;
    assert_eq!(dataset["status"], "ready");
    assert_eq!(dataset["stats"]["items"], 10);

    let (_, recs) = call(
        &app,
        post(
            "/api/v1/datasets/articles/recommend",
            json!({"query": "deep learning neural networks"}),
        ),
    )
    .await;
    assert_eq!(ids(&recs)[0], "102");

    let (status, item) = call(&app, get("/api/v1/datasets/articles/items/102")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(item["payload"]["topic"], "deep-learning");

    let resp = test::call_service(
        &app,
        TestRequest::delete()
            .uri("/api/v1/datasets/articles/items/102")
            .to_request(),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        call(&app, get("/api/v1/datasets/articles/items/102"))
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    let (_, recs) = call(
        &app,
        post(
            "/api/v1/datasets/articles/recommend",
            json!({"query": "deep learning neural networks"}),
        ),
    )
    .await;
    assert!(!ids(&recs).contains(&"102".to_string()));

    // A background upload answers 202 and becomes ready.
    let resp = test::call_service(
        &app,
        upload(
            "/api/v1/datasets/articles-bg/items/upload?name=Articles",
            "articles.json",
            ARTICLES_JSON.as_bytes(),
        ),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    assert_eq!(
        resp.headers().get("location").unwrap(),
        "/api/v1/datasets/articles-bg"
    );
    let ready = wait_ready(&app, "articles-bg").await;
    assert_eq!(ready["stats"]["items"], 10);
    assert_eq!(ready["name"], "Articles");

    let (_, list) = call(&app, get("/api/v1/datasets")).await;
    assert_eq!(list["datasets"].as_array().unwrap().len(), 2);
}

#[actix_web::test]
async fn parquet_and_ndjson_uploads() {
    let (state, dir) = state(|_| {}).await;
    let app = app!(state);

    let csv = dir.path().join("p.csv");
    std::fs::write(&csv, PRODUCTS_CSV).unwrap();
    let mut df = tabular::read_file(&csv, FileFormat::Csv).unwrap();
    let parquet = dir.path().join("p.parquet");
    tabular::write_parquet(&mut df, &parquet).unwrap();
    let (status, body) = call(
        &app,
        upload(
            "/api/v1/datasets/pq/items/upload?wait=true",
            "p.parquet",
            &std::fs::read(&parquet).unwrap(),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["dataset"]["stats"]["items"], 20);

    let ndjson =
        "{\"sku\": \"a\", \"title\": \"Red apple\"}\n{\"sku\": \"b\", \"title\": \"Green pear\"}\n";
    let (status, body) = call(
        &app,
        upload(
            "/api/v1/datasets/fruit/items/upload?wait=true",
            "fruit.ndjson",
            ndjson.as_bytes(),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["dataset"]["id_column"], "sku");
    assert_eq!(body["dataset"]["stats"]["items"], 2);
}

#[actix_web::test]
async fn interactions_personalize_recommendations() {
    let (state, _dir) = state(|_| {}).await;
    let app = app!(state);
    load_products(&app, "shop").await;
    let uri = "/api/v1/datasets/shop/recommend";

    let (status, body) = call(
        &app,
        post(
            "/api/v1/datasets/shop/interactions",
            json!({"interactions": [
                {"user_id": "u1", "item_id": "1", "event_type": "purchase", "weight": 3},
                {"user_id": "u1", "item_id": 16, "event_type": "view"},
                {"user_id": "u1", "item_id": "9"},
                {"user_id": "u1", "item_id": "999"}
            ]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["accepted"], 3);
    assert_eq!(body["users_updated"], 1);
    assert_eq!(body["unknown_items"], json!(["999"]));

    let (status, body) = call(&app, post(uri, json!({"user_id": "u1", "limit": 5}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let recs = ids(&body);
    assert_eq!(recs.len(), 5);
    assert!(
        recs.iter()
            .all(|id| !["1", "16", "9"].contains(&id.as_str())),
        "seen items are excluded: {recs:?}"
    );

    let (_, body) = call(
        &app,
        post(
            uri,
            json!({"user_id": "u1", "limit": 20, "exclude_seen": false}),
        ),
    )
    .await;
    assert!(ids(&body).contains(&"1".to_string()));

    // Unknown users fail unless another signal is present.
    assert_eq!(
        call(&app, post(uri, json!({"user_id": "nobody"}))).await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        call(
            &app,
            post(uri, json!({"user_id": "nobody", "query": "boots"}))
        )
        .await
        .0,
        StatusCode::OK
    );

    // Bulk interactions upload.
    let (status, body) = call(
        &app,
        upload(
            "/api/v1/datasets/shop/interactions/upload?wait=true",
            "events.csv",
            INTERACTIONS_CSV.as_bytes(),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["dataset"]["stats"]["users"].as_u64().unwrap() >= 3);

    let (status, _) = call(
        &app,
        post(
            "/api/v1/datasets/missing/interactions",
            json!({"interactions": [{"user_id": "u", "item_id": "1"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[actix_web::test]
async fn retrain_rebuilds_items_and_profiles() {
    let (state, _dir) = state(|_| {}).await;
    let app = app!(state);
    load_products(&app, "shop").await;
    call(&app, post("/api/v1/datasets/shop/items", json!({"items": [
        {"id": "21", "name": "Trail Running Shoes", "description": "Grippy shoes for muddy trails", "category": "footwear"}
    ]})))
    .await;
    call(
        &app,
        post(
            "/api/v1/datasets/shop/interactions",
            json!({"interactions": [{"user_id": "u1", "item_id": "21"}]}),
        ),
    )
    .await;
    let resp = test::call_service(
        &app,
        TestRequest::delete()
            .uri("/api/v1/datasets/shop/items/2")
            .to_request(),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    let (status, body) = call(
        &app,
        TestRequest::post()
            .uri("/api/v1/datasets/shop/train?wait=true&reset=true")
            .to_request(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["dataset"]["status"], "ready");
    assert_eq!(
        body["dataset"]["stats"]["items"], 20,
        "20 uploaded + 1 added - 1 deleted"
    );
    assert_eq!(body["dataset"]["stats"]["users"], 1);
    assert_eq!(
        call(&app, get("/api/v1/datasets/shop/items/2")).await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        call(&app, get("/api/v1/datasets/shop/items/21")).await.0,
        StatusCode::OK
    );

    let (status, body) = call(
        &app,
        post(
            "/api/v1/datasets/shop/recommend",
            json!({"user_id": "u1", "limit": 3}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(!ids(&body).contains(&"21".to_string()));
}

#[actix_web::test]
async fn predictions_are_logged_and_exported() {
    let (state, _dir) = state(|_| {}).await;
    let app = app!(state);
    load_products(&app, "shop").await;
    let (_, rec) = call(
        &app,
        post(
            "/api/v1/datasets/shop/recommend",
            json!({"query": "headphones"}),
        ),
    )
    .await;
    let prediction_id = rec["prediction_id"].as_str().unwrap().to_string();

    let mut found = false;
    for _ in 0..100 {
        let (_, list) = call(&app, get("/api/v1/predictions?dataset_id=shop")).await;
        if list["logs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|l| l["prediction_id"] == prediction_id)
        {
            assert_eq!(list["logs"][0]["query"], "headphones");
            found = true;
            break;
        }
        actix_web::rt::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(found, "prediction was not logged");

    for (format, content_type, prefix) in [
        ("csv", "text/csv", b"prediction_id".as_slice()),
        (
            "parquet",
            "application/vnd.apache.parquet",
            b"PAR1".as_slice(),
        ),
        ("jsonl", "application/x-ndjson", b"{".as_slice()),
    ] {
        let resp = test::call_service(
            &app,
            get(&format!("/api/v1/predictions/export?format={format}")),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get("content-type").unwrap(), content_type);
        assert!(
            resp.headers()
                .get("content-disposition")
                .unwrap()
                .to_str()
                .unwrap()
                .contains("attachment")
        );
        assert!(test::read_body(resp).await.starts_with(prefix), "{format}");
    }
    assert_eq!(
        call(&app, get("/api/v1/predictions?date=yesterday"))
            .await
            .0,
        StatusCode::BAD_REQUEST
    );
}

#[actix_web::test]
async fn validation_errors_are_reported() {
    let (state, _dir) = state(|_| {}).await;
    let app = app!(state);
    load_products(&app, "shop").await;
    let uri = "/api/v1/datasets/shop/recommend";

    let cases = [
        (post(uri, json!({})), StatusCode::BAD_REQUEST),
        (
            post(uri, json!({"query": "x", "filters": {"a": 1}})),
            StatusCode::BAD_REQUEST,
        ),
        (
            post(uri, json!({"query": "x", "limit": 0})),
            StatusCode::BAD_REQUEST,
        ),
        (
            post(uri, json!({"query": "x", "diversity": 2.0})),
            StatusCode::BAD_REQUEST,
        ),
        (
            post(
                uri,
                json!({"query": "x", "filter": {"must": [{"key": "price", "match": {"value": 9.99}}]}}),
            ),
            StatusCode::BAD_REQUEST,
        ),
        (
            post("/api/v1/datasets/nope/recommend", json!({"query": "x"})),
            StatusCode::NOT_FOUND,
        ),
        (
            post("/api/v1/datasets/bad%20id/recommend", json!({"query": "x"})),
            StatusCode::BAD_REQUEST,
        ),
        (
            post("/api/v1/datasets", json!({"id": "shop"})),
            StatusCode::CONFLICT,
        ),
        (
            post("/api/v1/datasets", json!({"id": "../etc"})),
            StatusCode::BAD_REQUEST,
        ),
        (
            post("/api/v1/datasets/shop/items", json!({"items": []})),
            StatusCode::BAD_REQUEST,
        ),
        (
            upload("/api/v1/datasets/x/items/upload", "notes.txt", b"hello"),
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
        ),
        (
            upload(
                "/api/v1/datasets/x/items/upload?text_columns=missing",
                "p.csv",
                PRODUCTS_CSV.as_bytes(),
            ),
            StatusCode::BAD_REQUEST,
        ),
        (
            upload("/api/v1/datasets/x/items/upload", "empty.csv", b""),
            StatusCode::BAD_REQUEST,
        ),
        (
            TestRequest::post()
                .uri(uri)
                .insert_header(("content-type", "application/json"))
                .set_payload("{not json")
                .to_request(),
            StatusCode::BAD_REQUEST,
        ),
    ];
    for (req, expected) in cases {
        let (status, body) = call(&app, req).await;
        assert_eq!(status, expected, "{body}");
        assert!(body["error"]["code"].is_string(), "error envelope: {body}");
    }
    // Failed uploads must not leave half-created datasets behind.
    assert_eq!(
        call(&app, get("/api/v1/datasets/x")).await.0,
        StatusCode::NOT_FOUND
    );

    let resp = test::call_service(
        &app,
        TestRequest::delete()
            .uri("/api/v1/datasets/shop")
            .to_request(),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        call(&app, post(uri, json!({"query": "x"}))).await.0,
        StatusCode::NOT_FOUND
    );
}

#[actix_web::test]
async fn api_key_protects_the_api() {
    let (state, _dir) = state(|s| s.server.api_key = Some("s3cret".into())).await;
    let app = app!(state);

    assert_eq!(call(&app, get("/health")).await.0, StatusCode::OK);
    let (status, body) = call(&app, get("/api/v1/datasets")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "unauthorized");
    let wrong = TestRequest::get()
        .uri("/api/v1/datasets")
        .insert_header(("x-api-key", "nope"))
        .to_request();
    assert_eq!(call(&app, wrong).await.0, StatusCode::UNAUTHORIZED);
    let header = TestRequest::get()
        .uri("/api/v1/datasets")
        .insert_header(("x-api-key", "s3cret"))
        .to_request();
    assert_eq!(call(&app, header).await.0, StatusCode::OK);
    let bearer = TestRequest::get()
        .uri("/api/v1/datasets")
        .insert_header(("authorization", "Bearer s3cret"))
        .to_request();
    assert_eq!(call(&app, bearer).await.0, StatusCode::OK);
}
