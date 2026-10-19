//! Identifier validation and deterministic vector-store point ids.
//!
//! Point ids are UUIDv5 hashes of `(dataset_id, item_id)`, which makes every
//! write idempotent: re-uploading an item overwrites it instead of duplicating it,
//! and an item can be fetched by its business id without a search.

use uuid::Uuid;

use crate::error::AppError;

pub const MAX_DATASET_ID_LEN: usize = 64;
pub const MAX_ENTITY_ID_LEN: usize = 256;

const ITEM_NAMESPACE: Uuid = Uuid::from_u128(0x6f1d_5a3e_8c2b_4e71_9a0d_3b5c_7e9f_1a24);
const USER_NAMESPACE: Uuid = Uuid::from_u128(0x2b8e_4c91_7d3a_4f56_8e1b_9c0d_5a6f_3e72);
const ROW_NAMESPACE: Uuid = Uuid::from_u128(0x9c4a_1e7b_3f2d_4a68_b5c9_0e1f_7d3a_6b84);

/// Dataset ids appear in URLs and file paths, so they are restricted to
/// `[A-Za-z0-9_-]`, must start with an alphanumeric character and be at most 64 chars.
pub fn validate_dataset_id(id: &str) -> Result<(), AppError> {
    let valid = !id.is_empty()
        && id.len() <= MAX_DATASET_ID_LEN
        && id.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if valid {
        Ok(())
    } else {
        Err(AppError::bad_request(format!(
            "invalid dataset id '{id}': use 1-{MAX_DATASET_ID_LEN} characters from [A-Za-z0-9_-], starting with a letter or digit"
        )))
    }
}

/// Item and user ids are free-form but must be non-empty and reasonably short.
pub fn validate_entity_id(kind: &str, id: &str) -> Result<(), AppError> {
    if id.trim().is_empty() {
        return Err(AppError::bad_request(format!(
            "{kind} id must not be empty"
        )));
    }
    if id.len() > MAX_ENTITY_ID_LEN {
        return Err(AppError::bad_request(format!(
            "{kind} id is longer than {MAX_ENTITY_ID_LEN} bytes"
        )));
    }
    Ok(())
}

/// Point id of an item inside the shared items collection.
pub fn item_point_id(dataset_id: &str, item_id: &str) -> Uuid {
    Uuid::new_v5(&ITEM_NAMESPACE, scoped(dataset_id, item_id).as_bytes())
}

/// Point id of a user profile inside the shared users collection.
pub fn user_point_id(dataset_id: &str, user_id: &str) -> Uuid {
    Uuid::new_v5(&USER_NAMESPACE, scoped(dataset_id, user_id).as_bytes())
}

/// Stable id for rows that have no id column: a hash of the row's content, so
/// re-uploading the same file is idempotent.
pub fn content_id(canonical_row: &str) -> String {
    Uuid::new_v5(&ROW_NAMESPACE, canonical_row.as_bytes())
        .simple()
        .to_string()
}

fn scoped(dataset_id: &str, id: &str) -> String {
    // U+001F (unit separator) cannot appear in a valid dataset id, so the pair is unambiguous.
    format!("{dataset_id}\u{1f}{id}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_safe_dataset_ids() {
        for id in ["products", "shop-1", "A_b-9", "0"] {
            assert!(validate_dataset_id(id).is_ok(), "{id}");
        }
    }

    #[test]
    fn rejects_unsafe_dataset_ids() {
        let long = "a".repeat(65);
        for id in [
            "",
            "../etc",
            "a/b",
            "-x",
            "_x",
            "with space",
            "ünï",
            long.as_str(),
        ] {
            assert!(validate_dataset_id(id).is_err(), "{id}");
        }
    }

    #[test]
    fn point_ids_are_deterministic_and_scoped() {
        assert_eq!(item_point_id("a", "1"), item_point_id("a", "1"));
        assert_ne!(item_point_id("a", "1"), item_point_id("b", "1"));
        assert_ne!(item_point_id("a", "1"), user_point_id("a", "1"));
        assert_ne!(item_point_id("a", "1b"), item_point_id("a1", "b"));
    }

    #[test]
    fn validates_entity_ids() {
        assert!(validate_entity_id("item", "sku-1").is_ok());
        assert!(validate_entity_id("item", "  ").is_err());
        assert!(validate_entity_id("user", &"x".repeat(257)).is_err());
    }
}
