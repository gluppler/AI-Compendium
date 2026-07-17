pub mod bad_likert;
pub mod crescendo;
pub mod linear;
pub mod sequential;
pub mod tree;

pub fn catalog() -> Vec<String> {
    vec![
        "bad_likert".into(),
        "crescendo".into(),
        "linear".into(),
        "sequential".into(),
        "tree".into(),
    ]
}
