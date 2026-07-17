#[derive(Debug, Clone, PartialEq)]
pub enum VerdictKind {
    Refusal,
    Cooperating,
    Evasive,
}

#[derive(Debug, Clone)]
pub struct JudgeVerdict {
    pub kind: VerdictKind,
    pub reason: String,
    pub score: Option<f32>,
}

pub enum Exploitability {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct AttackResult {
    pub attack_name: String,
    pub goal: String,
    pub prompt: String,
    pub success: bool,
    pub response: String,
    pub attempts: usize,
    pub judge_verdicts: Vec<JudgeVerdict>,
    pub history: Vec<Turn>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Strategy {
    Simple,
    Crescendo,
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::Simple => write!(f, "simple"),
            Strategy::Crescendo => write!(f, "crescendo"),
        }
    }
}

impl std::str::FromStr for Strategy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "simple" => Ok(Strategy::Simple),
            "crescendo" => Ok(Strategy::Crescendo),
            _ => Err(format!("Unknown strategy: {s}. Use 'simple' or 'crescendo'")),
        }
    }
}

pub struct TemplateInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub exploitability: Exploitability,
    pub multi_turn: bool,
}
