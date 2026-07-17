# Weaknesses That are Specific to AI/ML Technology

## [CWE-1039 - Inadequate Detection or Handling of Adversarial Input Perturbations in Automated Recognition Mechanism](https://cwe.mitre.org/data/definitions/1039.html) — The product uses an automated mechanism such as machine learning to recognize complex data inputs (e.g. image or audio) as a particular concept or category, but it does not properly detect or handle inputs that have been modified or constructed in a way that causes the mechanism to detect a different, incorrect concept.

## - [CWE-1427 - Improper Neutralization of Input Used for LLM Prompting](https://cwe.mitre.org/data/definitions/1427.html) — The product uses externally-provided data to build prompts provided to large language models (LLMs), but the way these prompts are constructed causes the LLM to fail to distinguish between user-supplied inputs and developer provided system directives.

## [CWE-1426 - Improper Validation of Generative AI Output](https://cwe.mitre.org/data/definitions/1426.html) — The product invokes a generative AI/ML component whose behaviors and outputs cannot be directly controlled, but the product does not validate or insufficiently validates the outputs to ensure that they align with the intended security, content, or privacy policy.

## - [CWE-1434 - Insecure Setting of Generative AI/ML Model Inference Parameters](https://cwe.mitre.org/data/definitions/1434.html) — The product has a component that relies on a generative AI/ML model configured with inference parameters that produce an unacceptably high rate of erroneous or unexpected outputs.

# General Software Weaknesses that Appear in Products that Use or Support AI/ML Technology

## [CWE-22 - Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')](https://cwe.mitre.org/data/definitions/22.html) — The product uses external input to construct a pathname that is intended to identify a file or directory that is located underneath a restricted parent directory, but the product does not properly neutralize special elements within the pathname that can cause the pathname to resolve to a location that is outside of the restricted directory.

## [CWE-77 - Improper Neutralization of Special Elements used in a Command ('Command Injection')](https://cwe.mitre.org/data/definitions/77.html) — The product constructs all or part of a command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended command when it is sent to a downstream component.

## - [CWE-78 - Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')](https://cwe.mitre.org/data/definitions/78.html) — The product constructs all or part of an OS command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended OS command when it is sent to a downstream component.

## [CWE-79 - Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')](https://cwe.mitre.org/data/definitions/79.html) — The product does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output that is used as a web page that is served to other users.

## [CWE-94 - Improper Control of Generation of Code ('Code Injection')](https://cwe.mitre.org/data/definitions/94.html) — The product constructs all or part of a code segment using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the syntax or behavior of the intended code segment.

## - [CWE-95 - Improper Neutralization of Directives in Dynamically Evaluated Code ('Eval Injection')](https://cwe.mitre.org/data/definitions/95.html) — The product receives input from an upstream component, but it does not neutralize or incorrectly neutralizes code syntax before using the input in a dynamic evaluation call (e.g. "eval").

## [CWE-116 - Improper Encoding or Escaping of Output](https://cwe.mitre.org/data/definitions/116.html) — The product prepares a structured message for communication with another component, but encoding or escaping of the data is either missing or done incorrectly. As a result, the intended structure of the message is not preserved.

## - [CWE-250 - Execution with Unnecessary Privileges](https://cwe.mitre.org/data/definitions/250.html) — The product performs an operation at a privilege level that is higher than the minimum level required, which creates new weaknesses or amplifies the consequences of other weaknesses.

## [CWE-434 - Unrestricted Upload of File with Dangerous Type](https://cwe.mitre.org/data/definitions/434.html) — The product allows the upload or transfer of dangerous file types that are automatically processed within its environment.

## - [CWE-502 - Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html) — The product deserializes untrusted data without sufficiently ensuring that the resulting data will be valid.

## [CWE-862 - Missing Authorization](https://cwe.mitre.org/data/definitions/862.html) — The product does not perform an authorization check when an actor attempts to access a resource or perform an action.

## - [CWE-918 - Server-Side Request Forgery (SSRF)](https://cwe.mitre.org/data/definitions/918.html) — The web server receives a URL or similar request from an upstream component and retrieves the contents of this URL, but it does not sufficiently ensure that the request is being sent to the expected destination.

## [CWE-1336 - Improper Neutralization of Special Elements Used in a Template Engine](https://cwe.mitre.org/data/definitions/1336.html) — The product uses a template engine to insert or process externally-influenced input, but it does not neutralize or incorrectly neutralizes special elements or syntax that can be interpreted as template expressions or other code directives when processed by the engine.

## [CWE-1426 - Improper Validation of Generative AI Output](https://cwe.mitre.org/data/definitions/1426.html) — The product invokes a generative AI/ML component whose behaviors and outputs cannot be directly controlled, but the product does not validate or insufficiently validates the outputs to ensure that they align with the intended security, content, or privacy policy.

## - [CWE-1427 - Improper Neutralization of Input Used for LLM Prompting](https://cwe.mitre.org/data/definitions/1427.html) — The product uses externally-provided data to build prompts provided to large language models (LLMs), but the way these prompts are constructed causes the LLM to fail to distinguish between user-supplied inputs and developer provided system directives.

## [CWE-1434 - Insecure Setting of Generative AI/ML Model Inference Parameters](https://cwe.mitre.org/data/definitions/1434.html) — The product has a component that relies on a generative AI/ML model configured with inference parameters that produce an unacceptably high rate of erroneous or unexpected outputs.
