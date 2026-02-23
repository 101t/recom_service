use actix_web::{delete, get, post, web, HttpResponse};
use serde_json::json;

use crate::qdrant::client::{CreateCollectionRequest, parse_distance};
use crate::AppState;

/// List all Qdrant collections
#[utoipa::path(
    get,
    path = "/api/v1/collections",
    tag = "Collections",
    responses(
        (status = 200, description = "List of collection names"),
        (status = 500, description = "Internal server error")
    )
)]
#[get("/collections")]
pub async fn list_collections(app_state: web::Data<AppState>) -> HttpResponse {
    match app_state.qdrant_client.list_collections().await {
        Ok(names) => HttpResponse::Ok().json(json!({ "collections": names })),
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to list collections: {}", e)
        })),
    }
}

/// Get detailed info about a specific collection
#[utoipa::path(
    get,
    path = "/api/v1/collections/{name}",
    tag = "Collections",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Collection details",
         body = crate::qdrant::client::CollectionDetails),
        (status = 500, description = "Internal server error")
    )
)]
#[get("/collections/{name}")]
pub async fn get_collection(
    path: web::Path<String>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    let name = path.into_inner();
    match app_state.qdrant_client.get_collection_info(&name).await {
        Ok(info) => HttpResponse::Ok().json(info),
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to get collection info: {}", e)
        })),
    }
}

/// Create a new Qdrant collection
#[utoipa::path(
    post,
    path = "/api/v1/collections",
    tag = "Collections",
    request_body = CreateCollectionRequest,
    responses(
        (status = 201, description = "Collection created"),
        (status = 500, description = "Internal server error")
    )
)]
#[post("/collections")]
pub async fn create_collection(
    req: web::Json<CreateCollectionRequest>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    let distance = parse_distance(req.distance.as_deref().unwrap_or("cosine"));

    match app_state
        .qdrant_client
        .create_collection_with_name(&req.name, req.vector_size, distance)
        .await
    {
        Ok(_) => HttpResponse::Created().json(json!({
            "status": "success",
            "message": format!("Collection '{}' created", req.name)
        })),
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to create collection: {}", e)
        })),
    }
}

/// Delete a Qdrant collection
#[utoipa::path(
    delete,
    path = "/api/v1/collections/{name}",
    tag = "Collections",
    params(
        ("name" = String, Path, description = "Collection name to delete")
    ),
    responses(
        (status = 200, description = "Collection deleted"),
        (status = 500, description = "Internal server error")
    )
)]
#[delete("/collections/{name}")]
pub async fn delete_collection(
    path: web::Path<String>,
    app_state: web::Data<AppState>,
) -> HttpResponse {
    let name = path.into_inner();
    match app_state.qdrant_client.delete_collection(&name).await {
        Ok(_) => HttpResponse::Ok().json(json!({
            "status": "success",
            "message": format!("Collection '{}' deleted", name)
        })),
        Err(e) => HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to delete collection: {}", e)
        })),
    }
}

#[cfg(test)]
mod tests {
    use crate::qdrant::client::{CollectionDetails, CreateCollectionRequest};

    #[test]
    fn test_create_collection_request_deserialization() {
        let json = r#"{"name": "products", "vector_size": 768, "distance": "cosine"}"#;
        let req: CreateCollectionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "products");
        assert_eq!(req.vector_size, 768);
        assert_eq!(req.distance.as_deref(), Some("cosine"));
    }

    #[test]
    fn test_create_collection_request_defaults() {
        let json = r#"{"name": "test", "vector_size": 384}"#;
        let req: CreateCollectionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "test");
        assert!(req.distance.is_none());
    }

    #[test]
    fn test_collection_details_serialization() {
        let details = CollectionDetails {
            name: "recommendations".to_string(),
            points_count: 5000,
            vectors_count: 5000,
            status: "green".to_string(),
        };
        let json_str = serde_json::to_string(&details).unwrap();
        assert!(json_str.contains("recommendations"));
        assert!(json_str.contains("5000"));
        assert!(json_str.contains("green"));
    }
}
