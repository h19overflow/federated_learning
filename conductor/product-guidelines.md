# Product Guidelines

## Documentation & Prose Style
- **Tone:** Formal and Academic. All documentation, including user-facing text, READMEs, and technical reports, must maintain a precise, objective, and well-structured tone suitable for a university-level thesis or technical publication.
- **Language:** English. Use standard professional terminology related to medical imaging, machine learning, and federated learning.
- **Structure:** Use clear headings, numbered lists for procedures, and provide theoretical context for technical implementation choices.

## Visual Identity & UX/UI Principles
- **Clinical & Data-Centric:** The interface must prioritize clarity and precision. Use a minimalist, high-contrast aesthetic suitable for medical environments. Data visualizations (charts, ROC curves, confusion matrices) should be the primary focus.
- **Real-time Feedback:** Incorporate interactive elements and live-updating visualizations (e.g., loss curves, progress bars) to provide immediate feedback on training and experiment status.
- **Modern & Sophisticated:** Adhere to Material Design principles to ensure a polished, professional, and contemporary look. Use subtle animations and a consistent color palette to enhance the user experience without causing distraction.

## Communication & Error Handling
- **Detailed Transparency:** Technical errors must be communicated comprehensively. Provide detailed logs, specific error codes, and, where appropriate for the technical user, stack traces.
- **Actionable Guidance:** Every error message should provide specific guidance or suggestions on how to resolve the underlying issue.
- **System Status:** Maintain high visibility of system state (e.g., connection status to the FL server, resource utilization, task progress) at all times.

## Privacy & Ethical AI Communication
- **Explicit Disclosures:** The application must prominently display notices explaining the federated learning process, emphasizing that patient data remains local and only model weights are transmitted.
- **Transparency & Disclaimers:** Clearly present model confidence scores and performance metrics. Include explicit disclaimers regarding the experimental nature of the system and its current unsuitability for direct clinical diagnostic use.
- **Educational Support:** Use contextual help, tooltips, and informational sections to educate users on the mechanics of privacy-preserving AI and the specific benefits of the implemented federated learning framework.
