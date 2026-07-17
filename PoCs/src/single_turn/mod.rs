pub mod deterministic;
pub mod adversarial_poetry;
pub mod authority_escalation;
pub mod context_flooding;
pub mod context_poisoning;
pub mod embedded_instruction_json;
pub mod emotional_manipulation;
pub mod goal_redirection;
pub mod gray_box;
pub mod input_bypass;
pub mod math_problem;
pub mod multilingual;
pub mod permission_escalation;
pub mod prompt_injection;
pub mod prompt_probing;
pub mod roleplay;
pub mod semantic_manipulation;
pub mod synthetic_context_injection;
pub mod system_override;

pub fn catalog() -> Vec<String> {
    vec![
        "adversarial_poetry".into(),
        "authority_escalation".into(),
        "base64".into(),
        "character_stream".into(),
        "context_flooding".into(),
        "context_poisoning".into(),
        "embedded_instruction_json".into(),
        "emotional_manipulation".into(),
        "goal_redirection".into(),
        "gray_box".into(),
        "input_bypass".into(),
        "leetspeak".into(),
        "math_problem".into(),
        "multilingual".into(),
        "permission_escalation".into(),
        "prompt_injection".into(),
        "prompt_probing".into(),
        "roleplay".into(),
        "rot13".into(),
        "semantic_manipulation".into(),
        "synthetic_context_injection".into(),
        "system_override".into(),
    ]
}
